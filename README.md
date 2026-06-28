```md
<div dir="rtl" align="right">

# پیش‌بینی قیمت خانه با یادگیری ماشین

این پروژه با هدف **پیش‌بینی قیمت فروش خانه** با استفاده از دیتاست **Ames Housing** پیاده‌سازی شده است.  
در این پروژه، فرایند کامل یک مسئله یادگیری ماشین رگرسیون از **تحلیل اکتشافی داده‌ها (EDA)** تا **پیش‌پردازش، آموزش مدل، ارزیابی، مقایسه مدل‌ها و ساختاردهی ماژولار کد** انجام شده است.

---

## نمای کلی پروژه

در این پروژه، ابتدا داده‌ها به صورت اکتشافی بررسی شدند تا ساختار کلی دیتاست، مقادیر گمشده، توزیع متغیر هدف و ارتباط بین ویژگی‌ها تحلیل شود. سپس یک Pipeline استاندارد برای پیش‌پردازش ویژگی‌های عددی و دسته‌ای طراحی شد و در ادامه چند مدل رگرسیونی مختلف آموزش داده شدند تا بهترین مدل برای پیش‌بینی قیمت خانه انتخاب شود.

مراحل اصلی پروژه شامل موارد زیر است:

- تحلیل اکتشافی داده‌ها (EDA)
- بررسی مقادیر گمشده
- تحلیل متغیر هدف `SalePrice`
- بررسی همبستگی ویژگی‌ها
- شناسایی داده‌های پرت
- پیش‌پردازش داده‌ها با Pipeline
- آموزش و ارزیابی چند مدل رگرسیون
- مقایسه عملکرد مدل‌ها
- استخراج Feature Importance برای بهترین مدل
- ماژولار کردن کدها و تبدیل نوت‌بوک‌ها به ساختار پروژه‌ای

---

## دیتاست

- **نام دیتاست:** Ames Housing Dataset
- **متغیر هدف:** `SalePrice`

مسیر فایل دیتاست:

```bash
data/raw/ames_housing.csv
```

---

## ساختار پروژه

```bash
project/
├── data/
│   ├── raw/
│   │   └── ames_housing.csv
│   └── processed/
│       └── README.md
│
├── models/
│   ├── linear_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   └── README.md
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison.png
│   │   └── xgboost_feature_importance.png
│   └── model_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
├── requirements-notebook.txt
├── .gitignore
└── README.md
```

---

## تحلیل اکتشافی داده‌ها

در مرحله EDA، موارد زیر بررسی شدند:

- ابعاد دیتاست و نوع داده‌ها
- بررسی اولیه ستون‌ها
- تحلیل مقادیر گمشده
- تحلیل توزیع متغیر هدف `SalePrice`
- تحلیل همبستگی بین ویژگی‌ها
- شناسایی داده‌های پرت

این مرحله در نوت‌بوک‌ها انجام شد و مبنایی برای طراحی مراحل بعدی پروژه بود.

---

## پیش‌پردازش داده‌ها

برای آماده‌سازی داده‌ها از `ColumnTransformer` استفاده شده است.

### ویژگی‌های عددی
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### ویژگی‌های دسته‌ای
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

پس از اعمال Pipeline:

- **ابعاد داده آموزش:** `(2344, 302)`
- **ابعاد داده تست:** `(586, 302)`

---

## مدل‌های آموزش‌داده‌شده

در این پروژه سه مدل رگرسیونی آموزش داده شدند:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

---

## نتایج مدل‌ها

| مدل | RMSE | MAE | R2 Score |
|------|------:|------:|------:|
| Linear Regression | 29635.36 | 16034.11 | 0.8905 |
| Random Forest Regressor | 26750.50 | 15866.22 | 0.9107 |
| XGBoost Regressor | 23286.03 | 14340.73 | 0.9324 |

### بهترین مدل
بهترین عملکرد مربوط به مدل **XGBoost Regressor** بوده است.

---

## اهمیت ویژگی‌ها

برای بهترین مدل یعنی XGBoost، تحلیل Feature Importance انجام شده و ۱۵ ویژگی برتر استخراج شده‌اند.

فایل خروجی نمودار:

```bash
reports/figures/xgboost_feature_importance.png
```

---

## ماژولار کردن پروژه

پس از پیاده‌سازی اولیه در نوت‌بوک‌ها، منطق پروژه به صورت ماژولار داخل پوشه `src/` منتقل شد تا ساختار نهایی پروژه تمیزتر، قابل نگهداری‌تر و قابل ارائه‌تر باشد.

### `src/preprocessing.py`
شامل توابع مربوط به:
- بارگذاری داده
- جداسازی ویژگی‌ها و متغیر هدف
- تقسیم داده به train و test
- ساخت preprocessor

### `src/train.py`
شامل توابع مربوط به:
- ساخت مدل
- آموزش و ارزیابی مدل
- آموزش همه مدل‌های پشتیبانی‌شده

### `src/evaluate.py`
شامل توابع مربوط به:
- ساخت جدول نتایج
- انتخاب بهترین مدل
- ذخیره گزارش مقایسه مدل‌ها
- رسم نمودار مقایسه مدل‌ها

### `main.py`
نقطه ورود اصلی پروژه است و وظایف زیر را انجام می‌دهد:

- بارگذاری و پیش‌پردازش داده‌ها
- آموزش تمام مدل‌ها
- ارزیابی و مقایسه نتایج
- ذخیره مدل‌های آموزش‌داده‌شده
- ذخیره گزارش‌ها و نمودارها

---

## نحوه نصب و اجرا

### 1) کلون کردن پروژه

```bash
git clone <repository-url>
cd ames-housing-ml
```

### 2) ساخت و فعال‌سازی محیط مجازی

#### ویندوز
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### لینوکس / مک
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) نصب وابستگی‌ها

برای اجرای اصلی پروژه:

```bash
pip install -r requirements.txt
```

برای کار با نوت‌بوک‌ها:

```bash
pip install -r requirements-notebook.txt
```

---

## اجرای پروژه

برای اجرای کل Pipeline:

```bash
python main.py
```

این دستور مراحل زیر را انجام می‌دهد:

- پیش‌پردازش داده‌ها
- آموزش مدل‌ها
- ارزیابی عملکرد مدل‌ها
- ذخیره مدل‌های نهایی
- تولید گزارش مقایسه
- ذخیره نمودارهای خروجی

---

## خروجی‌های پروژه

### مدل‌های ذخیره‌شده
- `models/linear_regression_pipeline.joblib`
- `models/random_forest_pipeline.joblib`
- `models/xgboost_pipeline.joblib`

### گزارش‌ها
- `reports/model_comparison.csv`

### نمودارها
- `reports/figures/model_comparison.png`
- `reports/figures/xgboost_feature_importance.png`

---

## تکنولوژی‌ها و کتابخانه‌های استفاده‌شده

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---


</div>
```