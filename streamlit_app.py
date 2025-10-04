import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Job Placement ML Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy imports - only import when needed
def lazy_import_ml():
    """Import ML libraries only when needed"""
    global train_test_split, cross_val_score, StratifiedKFold
    global LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
    global StandardScaler, confusion_matrix, classification_report
    global roc_curve, auc, precision_recall_curve, f1_score
    global accuracy_score, recall_score, precision_score
    
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_curve, auc,
        precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
    )

def lazy_import_plotly():
    """Import plotly only when needed"""
    global go, px, make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# CACHED FUNCTIONS
@st.cache_data(show_spinner="Loading data...")
def load_and_preprocess_data(file_bytes, target_bytes=None, demo_mode=False):
    """Load and preprocess the dataset - CACHED"""
    import io
    df = pd.read_excel(io.BytesIO(file_bytes), header=[0, 1])
    
    # Flatten multi-level columns
    df.columns = ['_'.join(col).strip() if col[1] != 'nan' else col[0].strip() 
                  for col in df.columns.values]
    
    # Remove extra rows
    df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Rename columns
    column_mapping = {
        'Participant_nan': 'Participant',
        'Cohort_nan': 'Cohort',
        'Stipend_nan': 'Stipend',
        'Internship _nan': 'Internship',
        'Microsoft Training Certfications _Excel': 'Excel_Cert',
        'Microsoft Training Certfications _Expert Excel': 'Excel_Expert_Cert',
        'Microsoft Training Certfications _Word': 'Word_Cert',
        'Microsoft Training Certfications _0utlook': 'Outlook_Cert',
        'Microsoft Training Certfications _PowerPoint': 'PowerPoint_Cert',
        "Add'l Training_ ": 'Additional_Training',
        'Childcare Expenses_ ': 'Childcare_Expenses',
        'Travel Expenses_ ': 'Travel_Expenses',
        'Household Expenses_ ': 'Household_Expenses',
        'Education_GED or HSG': 'Education_GED',
        'Education_Some College': 'Education_Some_College',
        'Education_BA or BS': 'Education_BA_BS',
        '# of Children_under 12': 'Num_Children'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Add target
    if target_bytes is not None:
        import io
        target_df = pd.read_csv(io.BytesIO(target_bytes))
        df['Got_Job'] = target_df['Got_Job']
    elif demo_mode:
        np.random.seed(42)
        prob = (
            0.3 + 
            df['Excel_Cert'].fillna(0) * 0.1 + 
            df['Excel_Expert_Cert'].fillna(0) * 0.15 + 
            df['Word_Cert'].fillna(0) * 0.05 + 
            df['Additional_Training'].fillna(0) * 0.12 + 
            df['Internship'].fillna(0) * 0.18 + 
            df['Education_BA_BS'].fillna(0) * 0.1
        ).clip(0, 0.95)
        df['Got_Job'] = np.random.binomial(1, prob)
    else:
        df['Got_Job'] = None
    
    return df

@st.cache_resource(show_spinner="Training models...")
def train_models_cached(_X_train, _X_test, _y_train, _y_test):
    """Train models - CACHED with @st.cache_resource for model objects"""
    lazy_import_ml()
    
    models = {}
    results = {}
    
    # Logistic Regression
    log_reg = LogisticRegression(
        C=1.0, penalty='l2', max_iter=1000, random_state=42, class_weight='balanced'
    )
    log_reg.fit(_X_train, _y_train)
    models['Logistic Regression'] = log_reg
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=4, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight='balanced'
    )
    rf.fit(_X_train, _y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(_X_train, _y_train)
    models['Gradient Boosting'] = gb
    
    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(_X_test)
        y_pred_proba = model.predict_proba(_X_test)[:, 1]
        
        results[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(_y_test, y_pred),
            'precision': precision_score(_y_test, y_pred, zero_division=0),
            'recall': recall_score(_y_test, y_pred, zero_division=0),
            'f1': f1_score(_y_test, y_pred, zero_division=0),
            'specificity': recall_score(_y_test, y_pred, pos_label=0, zero_division=0),
            'confusion_matrix': confusion_matrix(_y_test, y_pred)
        }
    
    return models, results

@st.cache_data
def create_roc_curve(_y_test, _results):
    """Create ROC curve - CACHED"""
    lazy_import_plotly()
    lazy_import_ml()
    
    fig = go.Figure()
    
    for name, result in _results.items():
        fpr, tpr, _ = roc_curve(_y_test, result['probabilities'])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f'{name} (AUC = {roc_auc:.3f})',
            mode='lines', line=dict(width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Random Classifier',
        mode='lines', line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    return fig

@st.cache_data
def create_confusion_matrix_plot(_cm, model_name):
    """Create confusion matrix - CACHED"""
    lazy_import_plotly()
    
    labels = ['No Job', 'Got Job']
    cm_percent = _cm.astype('float') / _cm.sum(axis=1)[:, np.newaxis] * 100
    
    text = [[f'{_cm[i][j]}<br>({cm_percent[i][j]:.1f}%)' 
             for j in range(len(_cm[0]))] 
            for i in range(len(_cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=_cm, x=labels, y=labels, text=text,
        texttemplate='%{text}', colorscale='Blues'
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted', yaxis_title='Actual', height=400
    )
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🎯 Job Placement ML Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### *Powered by Advanced Machine Learning*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Control Panel")
        
        mode = st.radio(
            "Select Mode:",
            ["📁 Upload Data", "🎮 Demo Mode"],
            help="Upload your own data or try the demo"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.3, 0.05)
        threshold = st.slider("Classification Threshold", 0.3, 0.7, 0.5, 0.05)
        
        st.markdown("---")
        model_choice = st.selectbox(
            "Primary Model:",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Data Upload", 
        "🔬 Analysis", 
        "🤖 Training", 
        "📊 Results"
    ])
    
    # Tab 1: Data Upload
    with tab1:
        st.markdown("## 📂 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Main Dataset")
            uploaded_file = st.file_uploader(
                "Upload Excel file",
                type=['xlsx', 'xls'],
                key="main_file"
            )
        
        with col2:
            st.markdown("### Target Variable")
            target_file = st.file_uploader(
                "Upload CSV (optional in Demo Mode)",
                type=['csv'],
                key="target_file"
            )
        
        if mode == "🎮 Demo Mode":
            st.info("📌 Demo Mode - Synthetic targets will be generated")
            demo_mode = True
        else:
            demo_mode = False
        
        if uploaded_file is not None:
            if st.button("🚀 Load Data", type="primary"):
                try:
                    # Read file bytes for caching
                    file_bytes = uploaded_file.read()
                    target_bytes = target_file.read() if target_file else None
                    
                    df = load_and_preprocess_data(file_bytes, target_bytes, demo_mode)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Loaded {len(df)} participants")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Participants", len(df))
                    with col2:
                        if df['Got_Job'].notna().any():
                            st.metric("Job Rate", f"{df['Got_Job'].mean()*100:.1f}%")
                    with col3:
                        st.metric("Features", len(df.columns) - 3)
                    with col4:
                        st.metric("Complete", df.notna().all(axis=1).sum())
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Upload the main dataset to begin")
    
    # Tab 2: Analysis
    with tab2:
        if not st.session_state.data_loaded:
            st.info("👈 Load data first")
        else:
            df = st.session_state.df
            
            st.markdown("## 🔍 Quick Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Dataset Preview")
                st.dataframe(df.head(10), height=300)
            
            with col2:
                st.markdown("### Statistics")
                st.write(df.describe().round(2))
            
            if df['Got_Job'].notna().any():
                st.markdown("### Feature Correlations")
                
                feature_cols = [col for col in df.columns 
                               if col not in ['Participant', 'Cohort', 'Got_Job']]
                
                correlations = df[feature_cols + ['Got_Job']].corr()['Got_Job'].drop('Got_Job').sort_values(ascending=False)
                
                st.bar_chart(correlations)
    
    # Tab 3: Training
    with tab3:
        if not st.session_state.data_loaded:
            st.info("👈 Load data first")
        elif st.session_state.df['Got_Job'].isna().all():
            st.warning("No target variable. Use Demo Mode or upload targets")
        else:
            df = st.session_state.df
            
            st.markdown("## 🤖 Model Training")
            
            if st.button("🎯 Train Models", type="primary"):
                lazy_import_ml()
                
                # Prepare data
                feature_cols = [col for col in df.columns 
                               if col not in ['Participant', 'Cohort', 'Got_Job']]
                X = df[feature_cols].values
                y = df['Got_Job'].values
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train (CACHED!)
                models, results = train_models_cached(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )
                
                # Save to session
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_cols
                st.session_state.scaler = scaler
                st.session_state.model_trained = True
                
                st.success("✅ Models trained!")
                
                # Show comparison
                comparison_data = []
                for name, result in results.items():
                    comparison_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.3f}",
                        'Precision': f"{result['precision']:.3f}",
                        'Recall': f"{result['recall']:.3f}",
                        'F1': f"{result['f1']:.3f}",
                        'Specificity': f"{result['specificity']:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                
                best_model = max(results.items(), 
                               key=lambda x: x[1]['specificity'])[0]
                st.info(f"🏆 Best for Specificity: {best_model}")
    
    # Tab 4: Results
    with tab4:
        if not st.session_state.model_trained:
            st.info("👈 Train models first")
        else:
            st.markdown("## 📈 Model Performance")
            
            results = st.session_state.results
            y_test = st.session_state.y_test
            
            # ROC Curve (lazy loaded)
            with st.spinner("Generating ROC curve..."):
                roc_fig = create_roc_curve(y_test, results)
                st.plotly_chart(roc_fig, use_container_width=True)
            
            # Selected model analysis
            st.markdown(f"### {model_choice} Details")
            
            selected_result = results[model_choice]
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_fig = create_confusion_matrix_plot(
                    selected_result['confusion_matrix'],
                    model_choice
                )
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Metrics")
                st.metric("Accuracy", f"{selected_result['accuracy']:.1%}")
                st.metric("Precision", f"{selected_result['precision']:.1%}")
                st.metric("Recall", f"{selected_result['recall']:.1%}")
                st.metric("Specificity", f"{selected_result['specificity']:.1%}")
                st.metric("F1-Score", f"{selected_result['f1']:.1%}")
            
            # Quick prediction
            st.markdown("### 🎯 Quick Prediction")
            
            if st.button("Test with Sample Data"):
                feature_names = st.session_state.feature_names
                
                # Create sample input (all 1s)
                sample = np.ones((1, len(feature_names)))
                sample_scaled = st.session_state.scaler.transform(sample)
                
                model = st.session_state.models[model_choice]
                prob = model.predict_proba(sample_scaled)[0][1]
                
                st.metric("Job Placement Probability", f"{prob:.1%}")
                
                if prob >= threshold:
                    st.success("✅ High probability of placement")
                else:
                    st.warning("⚠️ May need additional support")

if __name__ == "__main__":
    main()
