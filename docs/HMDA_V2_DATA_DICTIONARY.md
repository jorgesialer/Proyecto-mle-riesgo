# Diccionario de datos — HMDA Loan Approval V2

## Fuente y poblacion

El dataset se construye a partir del **HMDA 2023 One Year National
Loan-Level Dataset**, publicado por CFPB/FFIEC y congelado el 19 de mayo de
2025. La extraccion usa el endpoint oficial del Data Browser documentado en
<https://ffiec.cfpb.gov/documentation/api/data-browser/>.

Poblacion incluida: aplicaciones de 2023 con `action_taken` 1, 2 o 3, que no
son hipotecas inversas, son closed-end, no tienen finalidad principalmente
comercial y corresponden a propiedades de una a cuatro unidades. El target se
crea como `1/2 -> LoanApproved=1` y `3 -> LoanApproved=0`; `action_taken` se
descarta inmediatamente despues.

El momento conceptual de prediccion es inmediatamente anterior a la decision
final, cuando los insumos de underwriting ya estan disponibles. Esto permite
evaluar income, DTI, CLTV y property value como candidatos, pero no elimina la
obligacion de auditar su missingness por clase ni demuestra causalidad.

## Columnas finales

| Campo HMDA original | Nombre final | Significado | Tipo esperado | Rol | Tratamiento de missing | Observacion de leakage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `income` | `income` | Ingreso bruto anual reportado, en miles de USD en la publicacion HMDA. | Numerico | Predictor candidato | `NA`, `Exempt` y vacio a nulo. | Disponible en underwriting; auditar faltantes condicionados por decision y outliers. |
| `loan_amount` | `loan_amount` | Monto del prestamo solicitado/reportado, modificado por CFPB para privacidad. | Numerico | Predictor candidato | Marcadores no numericos a nulo. | Predecision; no equivale al monto finalmente desembolsado sin verificar el contexto de accion. |
| `loan_term` | `loan_term` | Plazo del prestamo en meses. | Numerico | Predictor candidato | `NA`, `Exempt` y vacio a nulo. | Termino propuesto predecision. |
| `loan_purpose` | `loan_purpose` | Proposito del prestamo segun codigos HMDA. | Categorico codificado | Predictor candidato | Marcadores explicitos a nulo; codigos validos se conservan. | Predecision. |
| `loan_type` | `loan_type` | Tipo de prestamo: convencional, FHA, VA o RHS/FSA. | Categorico codificado | Predictor candidato | Marcadores explicitos a nulo. | Predecision. |
| `lien_status` | `lien_status` | Posicion prevista del gravamen. | Categorico codificado | Predictor candidato | Marcadores explicitos a nulo. | Predecision. |
| `preapproval` | `preapproval` | Estado de solicitud de preaprobacion. | Categorico codificado | Predictor candidato | `1111`, `NA`, `Exempt` y vacio a nulo. | Puede segmentar procesos; revisar estabilidad entre entidades. |
| `debt_to_income_ratio` | `debt_to_income_ratio` | DTI reportado; la publicacion contiene valores y bandas. | Categorico/ordinal mixto | Predictor candidato | `NA`, `Exempt` y vacio a nulo; bandas se conservan. | Insumo de underwriting permitido; missingness puede depender del resultado. |
| `loan_to_value_ratio` | `combined_loan_to_value_ratio` | Combined loan-to-value ratio reportado. | Numerico | Predictor candidato | `NA`, `Exempt` y vacio a nulo. | Insumo de underwriting permitido; missingness puede revelar procesos de decision. |
| `property_value` | `property_value` | Valor de la propiedad reportado, modificado por CFPB para privacidad. | Numerico | Predictor candidato | `NA`, `Exempt` y vacio a nulo. | Insumo permitido; auditar disponibilidad diferencial por clase. |
| `occupancy_type` | `occupancy_type` | Uso previsto: residencia principal, segunda residencia o inversion. | Categorico codificado | Predictor candidato | Marcadores explicitos a nulo. | Predecision. |
| `construction_method` | `construction_method` | Vivienda site-built o manufactured home. | Categorico codificado | Predictor candidato | Marcadores explicitos a nulo. | Predecision. |
| `total_units` | `total_units` | Numero de unidades de la propiedad; limitado aqui a 1–4. | Categorico/entero | Predictor candidato | No se esperan nulos tras el filtro. | Tambien define la poblacion; señal valida pero de variacion reducida. |
| `submission_of_application` | `submission_of_application` | Si la solicitud fue presentada directamente a la institucion. | Categorico codificado | Predictor candidato | `1111`, `NA`, `Exempt` y vacio a nulo. | Predecision; posible proxy del canal/institucion. |
| `interest_only_payment` | `interest_only_payment` | Si contempla pagos solo de intereses. | Categorico binario | Predictor candidato | `1111`, `NA`, `Exempt` y vacio a nulo. | Termino contractual propuesto. |
| `balloon_payment` | `balloon_payment` | Si contempla balloon payment. | Categorico binario | Predictor candidato | `1111`, `NA`, `Exempt` y vacio a nulo. | Termino contractual propuesto. |
| `applicant_age` | `applicant_age` | Banda de edad del solicitante publicada por CFPB. | Categorico ordinal | Solo auditoria | `8888`, `NA`, `Exempt` y vacio a nulo. | Atributo protegido/proxy: prohibido como predictor. |
| `derived_race` | `derived_race` | Raza derivada por reglas de publicacion HMDA. | Categorico | Solo auditoria | Marcadores tecnicos a nulo; categorias de no disponibilidad se conservan. | Atributo protegido: prohibido como predictor. |
| `derived_ethnicity` | `derived_ethnicity` | Etnicidad derivada por reglas de publicacion HMDA. | Categorico | Solo auditoria | Marcadores tecnicos a nulo; categorias de no disponibilidad se conservan. | Atributo protegido: prohibido como predictor. |
| `derived_sex` | `derived_sex` | Sexo derivado por reglas de publicacion HMDA. | Categorico | Solo auditoria | Marcadores tecnicos a nulo; categorias de no disponibilidad se conservan. | Atributo protegido: prohibido como predictor. |
| `action_taken` | `LoanApproved` | Decision historica binaria: aprobada/originada frente a denegada. | Binario entero | Target | No admite nulos; otros estados se excluyen. | El campo fuente se elimina despues del mapeo y nunca se entrega como predictor. |

## Roles y guardrail de entrenamiento

`src.construir_dataset_v2.MODEL_PREDICTOR_COLUMNS` contiene exclusivamente los
16 predictores crudos finales. `obtener_predictores_entrenamiento()` selecciona esa lista de
forma explicita, excluyendo `LoanApproved` y las cuatro columnas `audit_only`.
La futura fase de entrenamiento V2 debe usar este contrato y no una seleccion
del tipo “todas las columnas excepto el target”.

## Variables retiradas tras la auditoria del sample

| Variable | Evidencia observada | Decision |
| :--- | :--- | :--- |
| `negative_amortization` | 50,000 valores con codigo `2`; varianza nula. | Eliminada. |
| `introductory_rate_period` | 93.214% de missingness, principalmente estructural. | Eliminada; no se crea indicador de ausencia. |
| `other_nonamortizing_features` | 44 positivos de 50,000 (0.088%). | Eliminada por varianza practicamente nula. |
| `total_units` | 868 registros multiunidad y cuatro categorias observadas. | Conservada como categorica; su utilidad se evaluara mediante CV. |

## Features derivadas para modelado

Estas features se calculan de forma determinista dentro del pipeline, antes de
la imputacion. Una division con denominador nulo o no positivo produce `NaN`,
que posteriormente se imputa usando exclusivamente training data.

| Feature derivada | Definicion | Tipo | Fuente cruda | Observacion |
| :--- | :--- | :--- | :--- | :--- |
| `loan_to_income` | `loan_amount / (income * 1000)` | Numerica | `loan_amount`, `income` | Income HMDA se publica en miles de USD. |
| `property_value_to_income` | `property_value / (income * 1000)` | Numerica | `property_value`, `income` | No se calcula cuando income es cero/negativo. |
| `loan_to_property_value` | `loan_amount / property_value` | Numerica | `loan_amount`, `property_value` | Complementa, pero no sustituye, el CLTV reportado. |
| `loan_term_years` | `loan_term / 12` | Numerica | `loan_term` | Reexpresa el plazo sin aprender parametros. |
| `dti_category` | Banda canonica de DTI | Categorica ordinal | `debt_to_income_ratio` | Evita one-hot de valores numericos y bandas mezclados. |
| `non_amortizing_feature_count` | Conteo de flags afirmativos entre interest-only y balloon payment | Numerica discreta | Dos flags contractuales | Queda nula si falta cualquiera de los dos flags. |

Las bandas canonicas de DTI son: `<20`, `20–29`, `30–35`, `36–42`,
`43–49`, `50–60` y `>60`. Las bandas ya publicadas por HMDA se mapean
directamente; los valores numericos 36–49 se agrupan en `36–42` o `43–49`.
No se inventan puntos medios.

## Enumeraciones HMDA usadas por XAI

La capa XAI traduce las siguientes enumeraciones usando exclusivamente el
**2023 Reportable HMDA Data: Regulatory and Reporting Overview Reference
Chart** de CFPB. El codigo original siempre se conserva como `raw_code`; un
codigo fuera de estas tablas recibe `category_label="unknown"` y no se infiere.

| Feature | Codigo | Etiqueta XAI oficial |
| :--- | :--- | :--- |
| `loan_purpose` | `1` | Home purchase |
| `loan_purpose` | `2` | Home improvement |
| `loan_purpose` | `31` | Refinancing |
| `loan_purpose` | `32` | Cash-out refinancing |
| `loan_purpose` | `4` | Other purpose |
| `loan_purpose` | `5` | Not applicable |
| `loan_type` | `1` | Conventional |
| `loan_type` | `2` | FHA |
| `loan_type` | `3` | VA |
| `loan_type` | `4` | USDA/RHS/FSA |
| `lien_status` | `1` | First lien |
| `lien_status` | `2` | Subordinate lien |
| `occupancy_type` | `1` | Principal residence |
| `occupancy_type` | `2` | Second residence |
| `occupancy_type` | `3` | Investment property |
| `construction_method` | `1` | Site-built |
| `construction_method` | `2` | Manufactured home |

Fuente oficial: <https://files.consumerfinance.gov/f/documents/cfpb_reportable-hmda-data_regulatory-and-reporting-overview-reference-chart_2023-02.pdf>.

El frame determinista posterior a feature engineering contiene 20 variables:
9 numericas y 11 categoricas. `loan_term` y `debt_to_income_ratio` crudas son
reemplazadas por `loan_term_years` y `dti_category`, respectivamente.

## Blacklist

Se excluyen identificadores (`lei`), geografia granular, atributos protegidos
crudos, razones de denegacion, resultados AUS, purchaser type, interest rate,
rate spread, costos/pricing, variables de venta y otros campos posteriores a
la decision. Los dos campos especializados de manufactured home tambien se
excluyen de esta version por decision de alcance.

## Notas de interpretacion

- HMDA describe decisiones historicas de instituciones; `LoanApproved` no es
  default, repago ni una politica normativa ideal.
- Los valores publicados han sido modificados por CFPB para proteger la
  privacidad. Income y loan amount pueden contener outliers sin limite superior.
- La ausencia de un valor puede ser estructural, exenta o estar asociada al
  flujo de decision. La metadata reporta missingness global y por clase.
- El missingness observado en rechazos frente a aprobaciones es: income
  2.382% vs. 1.088%; DTI 5.289% vs. 0.911%; CLTV 20.306% vs. 4.171%; y
  property value 8.632% vs. 0.842%. Se conservan temporalmente, sin crear
  indicadores de missingness, y deben someterse a pruebas de robustez.
- Las metricas de futuros modelos V2 no deben compararse directamente con V1:
  cambian poblacion, fuente, variables y proceso generador del target.
