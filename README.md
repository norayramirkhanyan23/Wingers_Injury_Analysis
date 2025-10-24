# ⚽ Football Injury Analysis — Predicting Long-Term Durability

**Author:** Norayr Amirkhanyan  
**Tools:** Python, pandas, matplotlib, seaborn, rich  
**Date:** October 2025  

---

## 📘 Overview
This project explores whether early career workload (matches before age 22) impacts long-term durability and injury risk among elite 21st-century wingers.  
Data was **manually collected from Transfermarkt** and includes metrics such as matches, injuries, missed days, and positions.  
The visuals illustrate key trends in player health and workload balance.

---

## 📊 Visuals
- 🟦 **Match Distribution:** Before vs After age 22  
- 🟪 **Injury Growth:** Injury increase with age  
- ⚫ **Workload vs Durability:** Matches vs missed days (RW/LW comparison)  
- 🔥 **Correlation Matrix:** Relationship between workload and injuries  
- ⭐ **Best of the Best:** Messi vs Ronaldo radar comparison  
- 🧠 **Lamine Yamal Projection:** Predicting future durability  

---

## 🧩 Key Insights
- Wingers show **4–5× higher injury growth** after 22.  
- **Strong correlation (≈0.87)** between missed days and post-22 injuries.  
- Ronaldo and Salah demonstrate **exceptional durability** despite high workloads.  
- **Lamine Yamal’s projection** suggests careful load management is crucial.

---

## 🛠️ Run It Yourself
```bash
git clone https://github.com/AmirkhanyanNorayr/Wingers_Injury.git
cd Wingers_Injury
pip install pandas matplotlib seaborn rich
python Analysis.py
