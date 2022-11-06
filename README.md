# 設計多層感知機類神經網路
## 程式執行說明
### 程式限制
本程式可執行多維的分類但是三維以上or多層感知機所畫出來的分類線將無法準確劃分出分類結果，只能依據顏色來判斷分類結果
Output層限制只能用1個node，但隱藏層無限制
### GUI介紹
![圖片](https://user-images.githubusercontent.com/43849007/200174507-90460b5d-3bb7-421d-a277-f3a4b7ac164e.png)
![圖片](https://user-images.githubusercontent.com/43849007/200174559-64096ffe-50f7-48a5-9092-bff0c7b0c4af.png)
## 程式碼簡介
程式具有4個python檔:windows.py algorithm.py main.py resultPlotDraw.py

- main.py:主程式，主要功能為實作gui之功能和負責執行主程式，會先顯示gui介面，當使用者按下Train時會將輸入資料傳給algorithm.py處理，接著會將algorithm輸出資料傳入resultPlotDraw.py畫出散點圖和分隔線

- Windows.py:用PyQt5 designer 架構的gui介面，其中gui的功能實作在main.py中，將介面和功能分離的目的為方便修改介面時不會改動到功能

- algorithm.py:讀入輸入檔案，分離train data and test data，並根據上面感知機的原理計算train 的weight、準確度、RMSE和test的準確度、RMSE

- resultPlotDraw.py:負責繪出散點圖，並將圖的座標軸限制在剛好可顯示所有data的範圍，之後再劃分隔線(分隔線可能不會顯示在散點圖上，代表分隔線並沒有經過座標軸的範圍)

### 收斂和分類的方式
![圖片](https://user-images.githubusercontent.com/43849007/200174688-418c879e-82d5-4551-a529-085aa563c7d3.png) \
- 藍色為分類時判斷的邊界
- 紅色為收斂時的目標
在收斂時model 會將不同類別朝著不同的收斂目標 Ex:第一類會越來越靠近0.25、第二類會越來越靠近0.75
當model predict 該點的value的時候，classify function 會去判斷0~1之間要分成幾個類別並算出類別的boundary來進行分類。
