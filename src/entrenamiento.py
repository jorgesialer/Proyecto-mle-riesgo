import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class EntrenadorModelo:
    def __init__(self, ruta_datos: str, ruta_modelo: str, ruta_scaler: str):
        self.ruta_datos = ruta_datos
        self.ruta_modelo = ruta_modelo
        self.ruta_scaler = ruta_scaler
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()

    def ejecutar(self):
        print("1. Cargando datos limpios...")
        df = pd.read_csv(self.ruta_datos)
        
        X = df.drop('LoanApproved', axis=1)
        y = df['LoanApproved']
        
        print("2. Dividiendo y Escalando datos...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        columnas_numericas = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience']
        X_train[columnas_numericas] = self.scaler.fit_transform(X_train[columnas_numericas])
        
        print("3. Entrenando Random Forest...")
        self.modelo.fit(X_train, y_train)
        
        print("4. Guardando artefactos (.pkl)...")
        with open(self.ruta_modelo, 'wb') as f:
            pickle.dump(self.modelo, f)
        with open(self.ruta_scaler, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print("Éxito: Modelo y Scaler guardados en la carpeta artifacts/")

if __name__ == "__main__":
    entrenador = EntrenadorModelo(
        ruta_datos='data/credit_risk_dataset_limpio.csv',
        ruta_modelo='artifacts/modelo_rf_final.pkl',
        ruta_scaler='artifacts/scaler.pkl'
    )
    entrenador.ejecutar()