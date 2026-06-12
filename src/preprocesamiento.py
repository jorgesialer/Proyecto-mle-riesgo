import pandas as pd
import os

class PreprocesadorDatos:
    def __init__(self, ruta_entrada: str, ruta_salida: str):
        self.ruta_entrada = ruta_entrada
        self.ruta_salida = ruta_salida
        self.df = None

    def ejecutar(self):
        print("1. Cargando datos crudos...")
        self.df = pd.read_csv(self.ruta_entrada)
        
        print("2. Limpiando nulos y negativos...")
        self.df['Income'] = self.df['Income'].abs().fillna(self.df['Income'].median())
        self.df['LoanAmount'] = self.df['LoanAmount'].abs()
        self.df['CreditScore'] = self.df['CreditScore'].fillna(self.df['CreditScore'].median())
        self.df['Education'] = self.df['Education'].fillna(self.df['Education'].mode()[0])
        
        print("3. Aplicando One-Hot Encoding...")
        columnas_cat = ['Gender', 'Education', 'City', 'EmploymentType']
        self.df = pd.get_dummies(self.df, columns=columnas_cat, drop_first=True)
        
        for col in self.df.select_dtypes(include=['bool']).columns:
            self.df[col] = self.df[col].astype(int)
            
        self.df.to_csv(self.ruta_salida, index=False)
        print(f"Éxito: Datos preprocesados guardados en {self.ruta_salida}")

if __name__ == "__main__":
    preprocesador = PreprocesadorDatos(
        ruta_entrada='data/loan_risk_prediction_dataset.csv',
        ruta_salida='data/credit_risk_dataset_limpio.csv'
    )
    preprocesador.ejecutar()