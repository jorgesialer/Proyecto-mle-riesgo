"""Pruebas del contrato de entrenamiento, inferencia y persistencia."""

import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.entrenamiento import validar_target
from src.prediccion import SimuladorInferencia
from src.preprocesamiento import (
    COLUMNAS_PREDICTORAS,
    crear_pipeline_baseline,
    validar_columnas_predictoras,
)


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]


class PipelineBaselineTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "Age": [25, 35, 45, 55, 30, 40, 50, 60],
                "Income": [-1000, 40000, 50000, 60000, 35000, 45000, 55000, 65000],
                "LoanAmount": [5000, -6000, 7000, 8000, 4500, 6500, 7500, 8500],
                "CreditScore": [520, 610, np.nan, 740, 580, 650, 700, 780],
                "YearsExperience": [2, 5, 10, 20, 4, 8, 15, 25],
                "Gender": ["Female", "Male", "Female", "Male"] * 2,
                "Education": ["Bachelors", np.nan, "Masters", "PhD"] * 2,
                "City": ["Lima", "Cusco", "Lima", "Cusco"] * 2,
                "EmploymentType": ["Salaried", "Self-Employed"] * 4,
            }
        )
        self.y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], name="LoanApproved")

    def test_pipeline_contiene_preprocesamiento_y_random_forest(self):
        pipeline = crear_pipeline_baseline()
        self.assertIsInstance(pipeline, Pipeline)
        self.assertNotIn("LoanApproved", COLUMNAS_PREDICTORAS)
        self.assertIsInstance(
            pipeline.named_steps["clasificador"], RandomForestClassifier
        )

    def test_encoder_no_elimina_categoria_y_admite_desconocidas(self):
        pipeline = crear_pipeline_baseline()
        pipeline.fit(self.X, self.y)
        encoder = (
            pipeline.named_steps["preprocesamiento"]
            .named_transformers_["categoricas"]
            .named_steps["codificar"]
        )
        self.assertIsNone(encoder.drop)

        perfil = self.X.iloc[[0]].copy()
        perfil.loc[:, "City"] = "Ciudad no vista"
        perfil.loc[:, "EmploymentType"] = "Tipo no visto"
        self.assertEqual(pipeline.predict(perfil).shape, (1,))
        self.assertEqual(pipeline.predict_proba(perfil).shape, (1, 2))

    def test_negativos_se_invalidan_antes_de_imputar(self):
        pipeline = crear_pipeline_baseline()
        pipeline.fit(self.X, self.y)
        transformados = pipeline.named_steps["preprocesamiento"].transform(self.X)
        if hasattr(transformados, "toarray"):
            transformados = transformados.toarray()
        self.assertFalse(np.isnan(transformados).any())
        self.assertGreaterEqual(transformados[:, 0].min(), 0)
        self.assertGreaterEqual(transformados[:, 1].min(), 0)

    def test_esquema_valido(self):
        validar_columnas_predictoras(self.X.columns)

    def test_esquema_rechaza_predictor_faltante(self):
        with self.assertRaisesRegex(ValueError, "faltan columnas.*Income"):
            validar_columnas_predictoras(self.X.drop(columns="Income").columns)

    def test_esquema_rechaza_predictor_inesperado(self):
        columnas = list(self.X.columns) + ["UnexpectedFeature"]
        with self.assertRaisesRegex(ValueError, "columnas inesperadas"):
            validar_columnas_predictoras(columnas)

    def test_inferencia_valida_esquema_antes_del_pipeline(self):
        simulador = object.__new__(SimuladorInferencia)
        perfil_incompleto = self.X.iloc[0].drop(labels="CreditScore").to_dict()
        with self.assertRaisesRegex(ValueError, "faltan columnas.*CreditScore"):
            simulador.predecir(perfil_incompleto)

    def test_target_rechaza_valores_nulos(self):
        target = pd.Series([0, 1, np.nan], name="LoanApproved")
        with self.assertRaisesRegex(ValueError, "no puede contener valores nulos"):
            validar_target(target)

    def test_target_rechaza_clases_fuera_del_contrato_binario(self):
        target = pd.Series([0, 1, 2], name="LoanApproved")
        with self.assertRaisesRegex(ValueError, "exclusivamente las clases 0 y 1"):
            validar_target(target)

    def test_pickle_portable_en_subproceso_limpio(self):
        pipeline = crear_pipeline_baseline()
        pipeline.fit(self.X, self.y)

        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "pipeline.pkl"
            with ruta.open("wb") as archivo:
                pickle.dump(pipeline, archivo)

            codigo = textwrap.dedent(
                """
                import pickle
                import sys
                import pandas as pd

                with open(sys.argv[1], "rb") as archivo:
                    pipeline = pickle.load(archivo)

                perfil = pd.DataFrame([{
                    "Age": 35,
                    "Income": 85000,
                    "LoanAmount": 20000,
                    "CreditScore": 750,
                    "YearsExperience": 8,
                    "Gender": "Male",
                    "Education": "Bachelors",
                    "City": "New York",
                    "EmploymentType": "Salaried",
                }])
                assert pipeline.predict(perfil).shape == (1,)
                assert pipeline.predict_proba(perfil).shape == (1, 2)
                assert pipeline.named_steps["preprocesamiento"].transformers_[0][1].steps[0][1].func.__module__ == "src.preprocesamiento"
                print("pickle_subprocess_ok")
                """
            )
            entorno = os.environ.copy()
            entorno.pop("PYTHONPATH", None)
            resultado = subprocess.run(
                [sys.executable, "-c", codigo, str(ruta)],
                cwd=RAIZ_REPOSITORIO,
                env=entorno,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(resultado.stdout.strip(), "pickle_subprocess_ok")


if __name__ == "__main__":
    unittest.main()
