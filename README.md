# 使用 PyTorch 進行圖像分類   

本項目展示了如何使用自定義卷積神經網絡（CNN）和 PyTorch 進行圖像分類。示例中使用的數據集是一個來自《航海王》動畫系列的圖像集。  

## 概述  
本項目包括以下步驟：  
1. 從 zip 文件中提取圖像。  
2. 創建自定義數據集類來加載圖像及其對應的標籤。  
3. 定義 CNN 模型進行圖像分類。  
4. 訓練並評估模型。   
5. 報告成果。  

## 數據集  
數據集來自《航海王》動畫系列的圖像，分為訓練集和測試集。
- train/：訓練集影像資料夾。  
- test/：測試集影像資料夾。  
- classnames.txt：類別名稱列表。

## 模型架構  
使用 PyTorch 構建 CNN 模型，包括兩個卷積層模組，後接一個全連接層進行分類。  
- 輸入影像尺寸：64x64  
- 輸出類別數量：3  
- 使用交叉熵損失函數進行訓練。  
- 使用Adam優化器進行參數優化。  

## 使用方法  
確保數據集 zip 文件（one_piece_mini.zip）在項目目錄中。  
運行腳本以提取數據集並訓練模型  

## 訓練與評估  
訓練和評估過程包括：  

1. 載入和預處理數據集。  
2. 定義 CNN 模型。  
3. 在訓練數據集上訓練模型。    
4. 在測試數據集上評估模型。    
5. 在腳本中調整訓練次數和其他超參數。  

## 結果
訓練後，模型會在測試集上進行性能評估。每個訓練次數都會顯示訓練和測試的準確率及損失。最終結果將顯示模型在測試數據集上的準確度。  
