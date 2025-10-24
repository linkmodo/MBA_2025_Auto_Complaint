# 🚗 Vehicle Complaint Market Basket Analysis

A comprehensive Streamlit dashboard for analyzing NHTSA vehicle complaint data using Market Basket Analysis (MBA) to discover patterns and associations between vehicle components.

## 📊 Features

### 1. **Market Basket Analysis**
- Discover component failure associations using the Apriori algorithm
- Interactive parameter tuning (support, confidence, lift)
- Visualize association rules with scatter plots
- Filter by manufacturer for targeted analysis

### 2. **Temporal Analysis**
- Analyze complaint trends by year, month, and weekday
- Multiple date column options:
  - `date_received`: When NHTSA received the complaint (Column 17 - LDATE)
  - `date_added`: When complaint was added to database (Column 16 - DATEA)
  - `fail_date`: Date of incident (Column 8 - FAILDATE)

### 3. **Component Analysis**
- Top component frequency distributions
- Drill-down into specific component details
- View associated vehicle information and descriptions

### 4. **Manufacturer Insights**
- Filter entire dashboard by manufacturer
- Top manufacturer complaint distributions
- Pie chart visualizations

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12 (Python 3.13 has limited pandas support)
- NHTSA Vehicle Complaint Data CSV file

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/linkmodo/MBA_2025_Auto_Complaint.git
cd MBA_2025_Auto_Complaint
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add your data file**
Place your `Vehicle Complaint Data.csv` in the project root directory.

### Running Locally

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
MBA_2025_Auto_Complaint/
├── dashboard.py              # Main Streamlit dashboard
├── data_processor.py         # Data processing and MBA logic
├── app.py                    # Entry point (alternative)
├── mba_help.py              # MBA metrics documentation
├── column description.txt    # NHTSA data column definitions
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

## 📋 Data Format

The application expects NHTSA complaint data with the following key columns:

| Column # | Name | Description |
|----------|------|-------------|
| 3 | MFR_NAME | Manufacturer's name |
| 4 | MAKETXT | Vehicle/equipment make |
| 5 | MODELTXT | Vehicle/equipment model |
| 6 | YEARTXT | Model year |
| 12 | COMPDESC | Component description |
| 16 | DATEA | Date added to file (YYYYMMDD) |
| 17 | LDATE | Date complaint received by NHTSA (YYYYMMDD) |
| 20 | CDESCR | Description of complaint |

See `column description.txt` for complete column definitions.

## 🔧 Configuration

### MBA Parameters

- **Minimum Support** (0.01-0.2): Frequency threshold for itemsets
- **Association Metric**: Choose from lift, confidence, or support
- **Minimum Threshold**: Minimum value for selected metric

### Performance Tips

- For large datasets, filter by manufacturer first
- Start with higher support values (0.05+) and decrease if needed
- Use chunked processing for files >100MB

## 📊 Understanding MBA Metrics

### Support
Frequency of itemset appearing in all transactions
- **High support** = Common combination
- **Low support** = Rare but potentially interesting

### Confidence
Likelihood of consequent given antecedent
- **Formula**: Confidence(X→Y) = Support(X ∪ Y) / Support(X)
- **High confidence** = Strong predictability

### Lift
Strength of association compared to random chance
- **Lift > 1** = Positive association
- **Lift = 1** = Independent items
- **Lift < 1** = Negative association

See `mba_help.py` for detailed metric explanations.

## 🌐 Deployment to Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `dashboard.py` as the main file
   - Deploy!

3. **Upload Data**
   - Since the CSV is large (76MB), upload it via Streamlit Cloud's file upload feature
   - Or use a cloud storage link (S3, Google Drive, etc.)

### Important Notes for Deployment
- The CSV file is gitignored due to size
- You'll need to provide the data file separately
- Consider using a smaller sample for demo purposes

## 🛠️ Technologies Used

- **Streamlit**: Interactive web dashboard
- **Pandas**: Data processing and analysis
- **mlxtend**: Apriori algorithm and association rules
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations

## 📝 Requirements

```
mlxtend==0.23.1
pandas>=2.2.0
streamlit==1.31.0
plotly==5.18.0
wordcloud==1.9.3
plotly-express==0.4.1
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**linkmodo**
- GitHub: [@linkmodo](https://github.com/linkmodo)
- Repository: [MBA_2025_Auto_Complaint](https://github.com/linkmodo/MBA_2025_Auto_Complaint)

## 🙏 Acknowledgments

- NHTSA for providing vehicle complaint data
- Streamlit team for the amazing framework
- mlxtend library for MBA implementation

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: This application is for educational and analytical purposes. Always verify findings with domain experts before making business decisions.
