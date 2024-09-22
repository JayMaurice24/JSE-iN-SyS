import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from data import *
from MLP import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate (yRef, yOutput):
    mae = mean_absolute_error(yRef, yOutput)
    rmse = np.sqrt(mean_squared_error(yRef, yOutput))
    # print(np.abs((yRef - yOutput) / yRef))
    mape = torch.mean(torch.abs((yRef - yOutput) / yRef)) * 100

    return mae, rmse, mape


def file_write(comp, out_string):
    try:
        with open(comp, 'w') as file:
            file.write(out_string)
        print(f"Content successfully written to {comp}")
    except Exception as e:
        print(f"An error occurred: {e}")
def main():
    singleModel = input("Do you want to train for multiple companies together [Y/N] ?: ")
    if singleModel == 'N':
        manualRun = input("Provide manual window sizes and horizon [Y/N] ?: ")
        outputString = "-------------------------------------------------------------------------------\n"
        if manualRun == 'Y':
            windowSize = input("Please enter the input window size: ")
            horizon = input("Please enter the horizon: ")
            windowSize = int(windowSize)
            horizon = int(horizon)
            # print(windowSize, horizon)
            dataFile = "JSE_clean_truncated.csv"
            company = input("Please enter the name of the company (The name should match the column name in the csv): ")
            companyData = read_company_data(dataFile, company)
            xData, yData = size_data(windowSize, horizon, companyData)
            xTrain, yTrain, xVal, yVal, xTest, yTest = split_data(xData, yData, 0.7, 0.3, 0)
            print(len(xTrain), len(yTrain))
            print(len(xVal), len(yVal))
            print(len(xTest), len(yTest))

            trainingSet = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
            validSet = TensorDataset(torch.Tensor(xVal), torch.Tensor(yVal))
            testSet = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

            batchSize = 25
            learningRate = 0.0001
            trainLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=False)
            validLoader = DataLoader(validSet, batch_size=batchSize, shuffle=False)
            testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

            MLPModel = MLP(windowSize, horizon)
            optimizer = optim.Adam(MLPModel.parameters(), lr=learningRate)
            criterion = nn.L1Loss()
            epochs = 150
            trainingLossArray = []
            print("----------------------  Training  ----------------------------")
            for i in range(0, epochs):
                trainingLoss = 0
                for a, (inputs, ref) in enumerate(trainLoader):
                    optimizer.zero_grad()
                    outputs = MLPModel(inputs)
                    loss = criterion(outputs, ref)
                    trainingLoss += loss.item()
                    loss.backward()
                    optimizer.step()
                    # EpochLoss.append(loss.detach())
                # TrainingLoss.append(torch.tensor(EpochLoss).mean())
                # EpochLoss = []
                avgTrainingLoss = trainingLoss / len(trainingSet)
                trainingLossArray.append(avgTrainingLoss)
                # OutputString = OutputString + "Epoch " + str(i+1) + " out of " + str(MaxEpochs) + ": Loss = " + str(round(TrainingLoss[i].item(), 4))
                # OutputString = OutputString + "\n\n"
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(i + 1, epochs, avgTrainingLoss))

            plt.plot(range(0, epochs, 1), trainingLossArray)
            plt.title("A graph showing the Training Loss pattern over the training epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Training Loss")
            plt.show()

            print("----------------------  Testing  ----------------------------")

            testLoss = 0
            totalMAE = 0
            totalRMSE = 0
            totalMAPE = 0
            MLPModel.eval()
            with torch.no_grad():
                for inputs, ref in testLoader:
                    # print("Input: ", inputs)
                    # print("Ref: ", ref)
                    outputs = MLPModel(inputs)
                    # print(len(outputs[0]))
                    # print("Output: ", outputs)
                    mae, rmse, mape = evaluate(ref, outputs)
                    totalMAE += mae
                    totalRMSE += rmse
                    totalMAPE += mape
                    loss = criterion(outputs, ref)
                    testLoss += loss.item()

            # print(testLoss)
            avgTestLoss = testLoss / (len(testSet))
            print(f'Test Loss: {avgTestLoss:.4f}')
            # mae = totalMAE/len(testLoader)
            # rmse = totalRMSE/len(testLoader)
            # mape = totalMAPE/len(testLoader)
            # mae = totalMAE/horizon
            # rmse = totalRMSE/horizon
            # mape = totalMAPE/horizon
            mae = totalMAE
            rmse = totalRMSE
            mape = totalMAPE
            print(f"Window size: {windowSize}, Horizon: {horizon}")
            print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%\n")
        elif manualRun == 'N':

            windowSizeList = [30, 60, 120]
            horizonList = [1, 2, 5, 10, 30]
            # companyList = ["ASPEN", "CAPITEC", "IMPLATS", "GROWPNT", "NORTHAM", "ANGGOLD", "BATS", "EXXARO", "WOOLIES",
            #            "NASPERS-N-", "CLICKS", "BIDVEST", "SANLAM", "REMGRO", "GFIELDS", "RICHEMONT", "STANBANK", "SHOPRIT",
            #            "INVLTD", "MONDIPLC", "INVPLC", "DISCOVERY","AMPLATS", "ANGLO", "FIRSTRAND", "NEDBANK", "SASOL",
            #            "SPAR", "VODACOM", "MTN_GROUP"]

            companyList = ["IMPLATS", "GROWPNT", "NORTHAM", "ANGGOLD", "BATS", "EXXARO", "WOOLIES",
                           "NASPERS-N-", "CLICKS", "BIDVEST", "SANLAM", "REMGRO", "GFIELDS", "RICHEMONT", "STANBANK",
                           "SHOPRIT", "INVLTD", "MONDIPLC", "INVPLC", "DISCOVERY", "AMPLATS", "ANGLO", "FIRSTRAND", "NEDBANK",
                           "SASOL", "SPAR", "VODACOM", "MTN_GROUP"]



            dataFile = "JSE_clean_truncated.csv"

            for company in companyList:
                outputString += ("                                Company = " + company + '\n')
                for windowSize in windowSizeList:
                    for horizon in horizonList:
                        outputString += "\n-------------------------------------------------------------------------------\n"
                        outputString += ("Window Size = " + str(windowSize) + "    Horizon = " + str(horizon) + "\n")
                        companyData = read_company_data(dataFile, company)
                        xData, yData = size_data(windowSize, horizon, companyData)
                        xTrain, yTrain, xVal, yVal, xTest, yTest = split_data(xData, yData, 0.8, 0.2, 0)
                        # print(len(xTrain), len(yTrain))
                        # print(len(xVal), len(yVal))
                        # print(len(xTest), len(yTest))
                        outputString += ("\nNumber of Training Data = " + str(len(xTrain)) + "    Number of Testing Data = "+ str(len(xTest)) + "\n" + "\n")

                        trainingSet = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
                        validSet = TensorDataset(torch.Tensor(xVal), torch.Tensor(yVal))
                        testSet = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

                        batchSize = 25
                        learningRate = 0.0001
                        outputString += ("Learning Rate = " + str(learningRate) + "\n")

                        trainLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=False)
                        validLoader = DataLoader(validSet, batch_size=batchSize, shuffle=False)
                        testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

                        MLPModel = MLP(windowSize, horizon)
                        optimizer = optim.Adam(MLPModel.parameters(), lr=learningRate)
                        criterion = nn.L1Loss()
                        epochs = 125
                        trainingLossArray = []
                        print(company)
                        print(windowSize, horizon)
                        print("----------------------  Training  ----------------------------")
                        for i in range(0, epochs):
                            trainingLoss = 0
                            for a, (inputs, ref) in enumerate(trainLoader):
                                optimizer.zero_grad()
                                outputs = MLPModel(inputs)
                                loss = criterion(outputs, ref)
                                trainingLoss += loss.item()
                                loss.backward()
                                optimizer.step()
                                # EpochLoss.append(loss.detach())
                            # TrainingLoss.append(torch.tensor(EpochLoss).mean())
                            # EpochLoss = []
                            avgTrainingLoss = trainingLoss / (len(trainingSet))
                            trainingLossArray.append(avgTrainingLoss)
                            # OutputString = OutputString + "Epoch " + str(i+1) + " out of " + str(MaxEpochs) + ": Loss = " + str(round(TrainingLoss[i].item(), 4))
                            # OutputString = OutputString + "\n\n"
                            print('Epoch [{}/{}], Loss: {:.4f}'
                                  .format(i + 1, epochs, avgTrainingLoss))

                        # plt.plot(range(0, epochs, 1), trainingLossArray)
                        # plt.title("A graph showing the Training Loss pattern over the training epochs")
                        # plt.xlabel("Epochs")
                        # plt.ylabel("Training Loss")
                        # plt.show()

                        print("----------------------  Testing  ----------------------------")

                        testLoss = 0
                        totalMAE = 0
                        totalRMSE = 0
                        totalMAPE = 0
                        MLPModel.eval()
                        with torch.no_grad():
                            for inputs, ref in testLoader:
                                # print("Input: ", inputs)
                                # print("Ref: ", ref)
                                outputs = MLPModel(inputs)
                                # print(len(outputs[0]))
                                # print("Output: ", outputs)
                                mae, rmse, mape = evaluate(ref, outputs)
                                totalMAE += mae
                                totalRMSE += rmse
                                totalMAPE += mape
                                loss = criterion(outputs, ref)
                                testLoss += loss.item()

                        # print(testLoss)
                        avgTestLoss = testLoss / (len(testSet))
                        print(f'Test Loss: {avgTestLoss:.4f}')
                        # mae = totalMAE/len(testLoader)
                        # rmse = totalRMSE/len(testLoader)
                        # mape = totalMAPE/len(testLoader)
                        # mae = totalMAE/horizon
                        # rmse = totalRMSE/horizon
                        # mape = totalMAPE/horizon
                        mae = round(totalMAE, 3)
                        rmse = round(totalRMSE, 3)
                        mape = round(totalMAPE.item(), 3)
                        outputString += ("RMSE = " + str(rmse) + "\n")
                        outputString += ("MAE = " + str(mae) + "\n")
                        outputString += ("MAPE = " + str(mape) + "%" + "\n")
                        print(f"Window size: {windowSize}, Horizon: {horizon}")
                        print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%\n")
                fileName = "Output/" + str(company)+".txt"
                file_write(fileName, outputString)
                outputString = "-------------------------------------------------------------------------------\n"
    elif singleModel == 'Y':
        manualRun = input("Provide manual window sizes and horizon [Y/N] ?: ")
        outputString = "-------------------------------------------------------------------------------\n"
        if manualRun == 'Y':
            windowSize = input("Please enter the input window size: ")
            horizon = input("Please enter the horizon: ")
            windowSize = int(windowSize)
            horizon = int(horizon)
            # print(windowSize, horizon)
            dataFile = "JSE_clean_truncated.csv"
            company = ""
            companyList = []
            while company != "exit":
                company = input("Please enter the name of the company (The name should match the column name in the csv. Please enter 'exit' when all companies have been entered.): ")
                if company != "exit":
                    companyList.append(company)
            xTrain = []
            yTrain = []
            xVal = []
            yVal = []
            xTest = []
            yTest = []
            for company in companyList:
                # outputString += ("                                Company = " + company + '\n')
                companyData = read_company_data(dataFile, company)
                xData, yData = size_data(windowSize, horizon, companyData)
                xTrainHold, yTrainHold, xValHold, yValHold, xTestHold, yTestHold = split_data(xData, yData, 0.8, 0.2, 0)
                xTrain = xTrain + xTrainHold
                yTrain = yTrain + yTrainHold
                xVal = xVal + xValHold
                yVal = yVal + yValHold
                xTest = xTest + xTestHold
                yTest = yTest + yTestHold
            # companyData = read_company_data(dataFile, company)
            # xData, yData = size_data(windowSize, horizon, companyData)
            # xTrain, yTrain, xVal, yVal, xTest, yTest = split_data(xData, yData, 0.7, 0.3, 0)
            print(len(xTrain), len(yTrain))
            print(len(xVal), len(yVal))
            print(len(xTest), len(yTest))

            trainingSet = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
            validSet = TensorDataset(torch.Tensor(xVal), torch.Tensor(yVal))
            testSet = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

            batchSize = 25
            learningRate = 0.0001
            trainLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=False)
            validLoader = DataLoader(validSet, batch_size=batchSize, shuffle=False)
            testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

            MLPModel = MLP(windowSize, horizon)
            optimizer = optim.Adam(MLPModel.parameters(), lr=learningRate)
            criterion = nn.L1Loss()
            epochs = 150
            trainingLossArray = []
            print("----------------------  Training  ----------------------------")
            for i in range(0, epochs):
                trainingLoss = 0
                for a, (inputs, ref) in enumerate(trainLoader):
                    optimizer.zero_grad()
                    outputs = MLPModel(inputs)
                    loss = criterion(outputs, ref)
                    trainingLoss += loss.item()
                    loss.backward()
                    optimizer.step()
                    # EpochLoss.append(loss.detach())
                # TrainingLoss.append(torch.tensor(EpochLoss).mean())
                # EpochLoss = []
                avgTrainingLoss = trainingLoss / len(trainingSet)
                trainingLossArray.append(avgTrainingLoss)
                # OutputString = OutputString + "Epoch " + str(i+1) + " out of " + str(MaxEpochs) + ": Loss = " + str(round(TrainingLoss[i].item(), 4))
                # OutputString = OutputString + "\n\n"
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(i + 1, epochs, avgTrainingLoss))

            plt.plot(range(0, epochs, 1), trainingLossArray)
            plt.title("A graph showing the Training Loss pattern over the training epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Training Loss")
            plt.show()

            print("----------------------  Testing  ----------------------------")

            testLoss = 0
            totalMAE = 0
            totalRMSE = 0
            totalMAPE = 0
            MLPModel.eval()
            with torch.no_grad():
                for inputs, ref in testLoader:
                    # print("Input: ", inputs)
                    # print("Ref: ", ref)
                    outputs = MLPModel(inputs)
                    # print(len(outputs[0]))
                    # print("Output: ", outputs)
                    mae, rmse, mape = evaluate(ref, outputs)
                    totalMAE += mae
                    totalRMSE += rmse
                    totalMAPE += mape
                    loss = criterion(outputs, ref)
                    testLoss += loss.item()

            # print(testLoss)
            avgTestLoss = testLoss / (len(testSet))
            print(f'Test Loss: {avgTestLoss:.4f}')
            # mae = totalMAE/len(testLoader)
            # rmse = totalRMSE/len(testLoader)
            # mape = totalMAPE/len(testLoader)
            # mae = totalMAE/horizon
            # rmse = totalRMSE/horizon
            # mape = totalMAPE/horizon
            mae = totalMAE
            rmse = totalRMSE
            mape = totalMAPE
            print(f"Window size: {windowSize}, Horizon: {horizon}")
            print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%\n")
        elif manualRun == 'N':

            windowSizeList = [30, 60, 120]
            horizonList = [1, 2, 5, 10, 30]
            companyList = ["ASPEN", "CAPITEC", "IMPLATS", "GROWPNT", "NORTHAM", "ANGGOLD", "BATS", "EXXARO", "WOOLIES",
                       "NASPERS-N-"]


            dataFile = "JSE_clean_truncated.csv"


            for windowSize in windowSizeList:
                for horizon in horizonList:
                    xTrain = []
                    yTrain = []
                    xVal = []
                    yVal = []
                    xTest = []
                    yTest = []
                    for company in companyList:
                        # outputString += ("                                Company = " + company + '\n')
                        companyData = read_company_data(dataFile, company)
                        xData, yData = size_data(windowSize, horizon, companyData)
                        xTrainHold, yTrainHold, xValHold, yValHold, xTestHold, yTestHold = split_data(xData, yData, 0.8, 0.2, 0)
                        xTrain = xTrain + xTrainHold
                        yTrain = yTrain + yTrainHold
                        xVal = xVal + xValHold
                        yVal = yVal + yValHold
                        xTest = xTest + xTestHold
                        yTest = yTest + yTestHold

                    # print(len(xTrain), len(yTrain))
                    # print(len(xVal), len(yVal))
                    # print(len(xTest), len(yTest))
                    outputString += ("Window Size = " + str(windowSize) + "    Horizon = " + str(horizon) + "\n")
                    outputString += ("\nNumber of Training Data = " + str(
                        len(xTrain)) + "    Number of Testing Data = " + str(len(xTest)) + "\n")


                    trainingSet = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
                    validSet = TensorDataset(torch.Tensor(xVal), torch.Tensor(yVal))
                    testSet = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))
                    batchSize = 25
                    trainLoader = DataLoader(trainingSet, batch_size=batchSize, shuffle=False)
                    validLoader = DataLoader(validSet, batch_size=batchSize, shuffle=False)
                    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)


                    learningRate = 0.0000111
                    outputString += ("\nLearning Rate = " + str(learningRate) + "\n")
                    outputString += "-------------------------------------------------------------------------------\n"

                    MLPModel = MLP(windowSize, horizon)
                    optimizer = optim.Adam(MLPModel.parameters(), lr=learningRate)
                    criterion = nn.L1Loss()
                    epochs = 150
                    trainingLossArray = []
                    print(windowSize, horizon)
                    print("----------------------  Training  ----------------------------")
                    for i in range(0, epochs):
                        trainingLoss = 0
                        for a, (inputs, ref) in enumerate(trainLoader):
                            optimizer.zero_grad()
                            outputs = MLPModel(inputs)
                            loss = criterion(outputs, ref)
                            trainingLoss += loss.item()
                            loss.backward()
                            optimizer.step()
                            # EpochLoss.append(loss.detach())
                        # TrainingLoss.append(torch.tensor(EpochLoss).mean())
                        # EpochLoss = []
                        avgTrainingLoss = trainingLoss / (len(trainingSet))
                        trainingLossArray.append(avgTrainingLoss)
                        # OutputString = OutputString + "Epoch " + str(i+1) + " out of " + str(MaxEpochs) + ": Loss = " + str(round(TrainingLoss[i].item(), 4))
                        # OutputString = OutputString + "\n\n"
                        print('Epoch [{}/{}], Loss: {:.4f}'
                              .format(i + 1, epochs, avgTrainingLoss))

                    # plt.plot(range(0, epochs, 1), trainingLossArray)
                    # plt.title("A graph showing the Training Loss pattern over the training epochs")
                    # plt.xlabel("Epochs")
                    # plt.ylabel("Training Loss")
                    # plt.show()

                    print("----------------------  Testing  ----------------------------")

                    testLoss = 0
                    totalMAE = 0
                    totalRMSE = 0
                    totalMAPE = 0
                    MLPModel.eval()
                    with torch.no_grad():
                        for inputs, ref in testLoader:
                            # print("Input: ", inputs)
                            # print("Ref: ", ref)
                            outputs = MLPModel(inputs)
                            # print(len(outputs[0]))
                            # print("Output: ", outputs)
                            mae, rmse, mape = evaluate(ref, outputs)
                            totalMAE += mae
                            totalRMSE += rmse
                            totalMAPE += mape
                            loss = criterion(outputs, ref)
                            testLoss += loss.item()

                    # print(testLoss)
                    avgTestLoss = testLoss / (len(testSet))
                    print(f'Test Loss: {avgTestLoss:.4f}')
                    # mae = totalMAE/len(testLoader)
                    # rmse = totalRMSE/len(testLoader)
                    # mape = totalMAPE/len(testLoader)
                    # mae = totalMAE/horizon
                    # rmse = totalRMSE/horizon
                    # mape = totalMAPE/horizon
                    mae = round(totalMAE, 3)
                    rmse = round(totalRMSE, 3)
                    mape = round(totalMAPE.item(), 3)
                    outputString += ("RMSE = " + str(rmse) + "\n")
                    outputString += ("MAE = " + str(mae) + "\n")
                    outputString += ("MAPE = " + str(mape) + "%" + "\n")
                    print(f"Window size: {windowSize}, Horizon: {horizon}")
                    print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%\n")
                    fileName = r"Output for Combined\Window Size " + str(windowSize) + " Horizon " + str(horizon) + ".txt"
                    print(fileName)
                    file_write(fileName, outputString)
                    outputString = "-------------------------------------------------------------------------------\n"




if __name__ == "__main__":
    main()

