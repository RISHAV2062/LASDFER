import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Job Placement ML Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Helper Functions
@st.cache_data
def load_and_preprocess_data(file, target_file=None, demo_mode=False):
    """Load and preprocess the dataset"""
    df = pd.read_excel(file, header=[0, 1])
    
    # Flatten multi-level columns
    df.columns = ['_'.join(col).strip() if col[1] != 'nan' else col[0].strip() 
                  for col in df.columns.values]
    
    # Remove extra rows (header rows that became data)
    df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with all NaN
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Rename columns for clarity
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
    
    # Add target variable
    if target_file is not None:
        target_df = pd.read_csv(target_file)
        df['Got_Job'] = target_df['Got_Job']
    elif demo_mode:
        # Generate realistic demo targets based on features
        np.random.seed(42)
        # Create a weighted probability based on certifications and training
        prob = (
            0.3 + 
            df['Excel_Cert'] * 0.1 + 
            df['Excel_Expert_Cert'] * 0.15 + 
            df['Word_Cert'] * 0.05 + 
            df['Additional_Training'] * 0.12 + 
            df['Internship'] * 0.18 + 
            (df['Education_BA_BS']) * 0.1
        ).clip(0, 0.95)
        df['Got_Job'] = np.random.binomial(1, prob)
    else:
        df['Got_Job'] = None
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models"""
    models = {}
    results = {}
    
    # Logistic Regression with L2 regularization
    log_reg = LogisticRegression(
        C=1.0, 
        penalty='l2', 
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = log_reg
    
    # Random Forest with careful parameters to avoid overfitting
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # Evaluate models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return models, results

def create_roc_curve(y_test, results):
    """Create interactive ROC curve"""
    fig = go.Figure()
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {roc_auc:.3f})',
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        height=500
    )
    
    return fig

def create_feature_importance(model, feature_names, model_name):
    """Create feature importance visualization"""
    if model_name == 'Logistic Regression':
        importance = np.abs(model.coef_[0])
        title = 'Feature Importance (Absolute Coefficients)'
    else:
        importance = model.feature_importances_
        title = 'Feature Importance'
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    fig = go.Figure(go.Bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importance[indices],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=f'{title} - {model_name}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        showlegend=False
    )
    
    return fig

def create_confusion_matrix_plot(cm, model_name):
    """Create interactive confusion matrix"""
    labels = ['No Job', 'Got Job']
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    text = [[f'{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)' 
             for j in range(len(cm[0]))] 
            for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig

def interpret_model(model, feature_names, model_name):
    """Generate human-readable interpretation"""
    interpretation = f"## 🔍 {model_name} Interpretation\n\n"
    
    if model_name == 'Logistic Regression':
        coefs = model.coef_[0]
        # Get odds ratios
        odds_ratios = np.exp(coefs)
        
        # Sort by absolute coefficient value
        indices = np.argsort(np.abs(coefs))[::-1]
        
        interpretation += "### Key Insights:\n\n"
        interpretation += "**Positive Predictors (Increase job placement probability):**\n\n"
        
        positive_features = [(feature_names[i], coefs[i], odds_ratios[i]) 
                           for i in indices if coefs[i] > 0.1][:5]
        
        if positive_features:
            for feat, coef, odds in positive_features:
                impact = (odds - 1) * 100
                interpretation += f"- **{feat}**: Increases odds by {impact:.1f}% (coefficient: {coef:.3f})\n"
        
        interpretation += "\n**Negative Predictors (Decrease job placement probability):**\n\n"
        
        negative_features = [(feature_names[i], coefs[i], odds_ratios[i]) 
                           for i in indices if coefs[i] < -0.1][:5]
        
        if negative_features:
            for feat, coef, odds in negative_features:
                impact = (1 - odds) * 100
                interpretation += f"- **{feat}**: Decreases odds by {impact:.1f}% (coefficient: {coef:.3f})\n"
        
    else:
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        interpretation += "### Top 5 Most Important Features:\n\n"
        for i, idx in enumerate(indices[:5], 1):
            interpretation += f"{i}. **{feature_names[idx]}**: {importance[idx]:.4f}\n"
    
    return interpretation

# Main App
def main():
    st.markdown('<h1 class="main-header">🎯 Job Placement ML Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### *Powered by Advanced Machine Learning*")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
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
        st.markdown("### 📈 Model Selection")
        model_choice = st.selectbox(
            "Primary Model:",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Data Upload", 
        "🔬 Exploratory Analysis", 
        "🤖 Model Training", 
        "📊 Results & Insights",
        "🎯 Make Predictions"
    ])
    
    # Tab 1: Data Upload
    with tab1:
        st.markdown("## 📂 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Main Dataset")
            uploaded_file = st.file_uploader(
                "Upload Excel file with features",
                type=['xlsx', 'xls'],
                help="Upload the transformation squared data file"
            )
        
        with col2:
            st.markdown("### Target Variable (Optional)")
            target_file = st.file_uploader(
                "Upload CSV with 'Got_Job' column",
                type=['csv'],
                help="CSV with participant IDs and job outcomes"
            )
        
        if mode == "🎮 Demo Mode":
            st.info("📌 Running in **Demo Mode** - Synthetic targets will be generated for demonstration")
            demo_mode = True
            if uploaded_file is None:
                st.warning("Please upload the main dataset to proceed")
                return
        else:
            demo_mode = False
            if uploaded_file is None:
                st.warning("Please upload the main dataset")
                return
            if target_file is None:
                st.warning("Please upload the target variable file, or switch to Demo Mode")
                return
        
        if st.button("🚀 Load & Process Data", type="primary"):
            with st.spinner("Processing data..."):
                try:
                    df = load_and_preprocess_data(uploaded_file, target_file, demo_mode)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully! {len(df)} participants")
                    
                    # Quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Participants", len(df))
                    with col2:
                        if df['Got_Job'].notna().any():
                            st.metric("Job Placement Rate", f"{df['Got_Job'].mean()*100:.1f}%")
                    with col3:
                        st.metric("Features", len(df.columns) - 3)
                    with col4:
                        st.metric("Complete Records", df.notna().all(axis=1).sum())
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    # Tab 2: Exploratory Analysis
    with tab2:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first from the Data Upload tab")
        else:
            df = st.session_state.df
            
            st.markdown("## 🔍 Data Exploration")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            with col2:
                st.markdown("### 📊 Summary Statistics")
                st.write(df.describe())
            
            # Feature distributions
            st.markdown("### 📈 Feature Distributions")
            
            feature_cols = [col for col in df.columns if col not in ['Participant', 'Cohort', 'Got_Job']]
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                feature_cols,
                default=feature_cols[:4]
            )
            
            if selected_features and df['Got_Job'].notna().any():
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_features[:4]
                )
                
                for idx, feature in enumerate(selected_features[:4], 1):
                    row = (idx - 1) // 2 + 1
                    col = (idx - 1) % 2 + 1
                    
                    data_no_job = df[df['Got_Job'] == 0][feature]
                    data_got_job = df[df['Got_Job'] == 1][feature]
                    
                    fig.add_trace(
                        go.Histogram(x=data_no_job, name='No Job', opacity=0.7),
                        row=row, col=col
                    )
                    fig.add_trace(
                        go.Histogram(x=data_got_job, name='Got Job', opacity=0.7),
                        row=row, col=col
                    )
                
                fig.update_layout(height=600, showlegend=True, barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            if df['Got_Job'].notna().any():
                st.markdown("### 🔗 Feature Correlations with Job Outcome")
                
                correlations = df[feature_cols + ['Got_Job']].corr()['Got_Job'].drop('Got_Job').sort_values(ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker=dict(
                        color=correlations.values,
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title='Correlation with Job Placement',
                    xaxis_title='Correlation Coefficient',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Training
    with tab3:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first from the Data Upload tab")
        elif st.session_state.df['Got_Job'].isna().all():
            st.warning("No target variable available. Please upload target data or use Demo Mode")
        else:
            df = st.session_state.df
            
            st.markdown("## 🤖 Model Training Pipeline")
            
            if st.button("🎯 Train Models", type="primary"):
                with st.spinner("Training multiple models..."):
                    # Prepare data
                    feature_cols = [col for col in df.columns if col not in ['Participant', 'Cohort', 'Got_Job']]
                    X = df[feature_cols].values
                    y = df['Got_Job'].values
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                    
                    # Save to session state
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_cols
                    st.session_state.scaler = scaler
                    st.session_state.model_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display results
                    st.markdown("### 📊 Model Performance Comparison")
                    
                    comparison_data = []
                    for name, result in results.items():
                        comparison_data.append({
                            'Model': name,
                            'Accuracy': f"{result['accuracy']:.3f}",
                            'Precision': f"{result['precision']:.3f}",
                            'Recall': f"{result['recall']:.3f}",
                            'F1-Score': f"{result['f1']:.3f}",
                            'Specificity': f"{result['specificity']:.3f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Highlight best model for specificity
                    best_model = max(results.items(), key=lambda x: x[1]['specificity'])[0]
                    st.info(f"🏆 **Best Model for Specificity**: {best_model}")
                    
                    # Cross-validation scores
                    st.markdown("### 🔄 Cross-Validation Scores (5-Fold)")
                    
                    cv_scores = {}
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    
                    for name, model in models.items():
                        scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
                        cv_scores[name] = scores
                    
                    cv_data = []
                    for name, scores in cv_scores.items():
                        cv_data.append({
                            'Model': name,
                            'Mean CV Score': f"{scores.mean():.3f}",
                            'Std Dev': f"{scores.std():.3f}",
                            'Min': f"{scores.min():.3f}",
                            'Max': f"{scores.max():.3f}"
                        })
                    
                    cv_df = pd.DataFrame(cv_data)
                    st.dataframe(cv_df, use_container_width=True)
    
    # Tab 4: Results & Insights
    with tab4:
        if not st.session_state.model_trained:
            st.info("👈 Please train models first from the Model Training tab")
        else:
            st.markdown("## 📈 Comprehensive Model Analysis")
            
            results = st.session_state.results
            y_test = st.session_state.y_test
            
            # ROC Curves
            st.markdown("### 📉 ROC Curves")
            roc_fig = create_roc_curve(y_test, results)
            st.plotly_chart(roc_fig, use_container_width=True)
            
            # Model-specific analysis
            st.markdown(f"### 🔍 Detailed Analysis: {model_choice}")
            
            selected_result = results[model_choice]
            selected_model = st.session_state.models[model_choice]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm_fig = create_confusion_matrix_plot(
                    selected_result['confusion_matrix'],
                    model_choice
                )
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                # Metrics
                st.markdown("#### 📊 Performance Metrics")
                metrics_html = f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; color: white;">
                    <h3 style="color: white; margin-top: 0;">Key Metrics</h3>
                    <p><strong>Accuracy:</strong> {selected_result['accuracy']:.1%}</p>
                    <p><strong>Precision:</strong> {selected_result['precision']:.1%}</p>
                    <p><strong>Recall (Sensitivity):</strong> {selected_result['recall']:.1%}</p>
                    <p><strong>Specificity:</strong> {selected_result['specificity']:.1%}</p>
                    <p><strong>F1-Score:</strong> {selected_result['f1']:.1%}</p>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Classification Report
                st.markdown("#### 📝 Classification Report")
                report = classification_report(
                    y_test, 
                    selected_result['predictions'],
                    target_names=['No Job', 'Got Job'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            # Feature Importance
            st.markdown("### ⭐ Feature Importance Analysis")
            importance_fig = create_feature_importance(
                selected_model,
                st.session_state.feature_names,
                model_choice
            )
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Model Interpretation
            st.markdown("### 💡 Model Interpretation & Insights")
            interpretation = interpret_model(
                selected_model,
                st.session_state.feature_names,
                model_choice
            )
            st.markdown(interpretation)
            
            # Additional insights
            st.markdown("### 🎯 Actionable Recommendations")
            
            if model_choice == 'Logistic Regression':
                coefs = selected_model.coef_[0]
                feature_names = st.session_state.feature_names
                
                # Top positive features
                top_positive_idx = np.argsort(coefs)[-3:][::-1]
                
                st.markdown("**To improve job placement outcomes, prioritize:**")
                for idx in top_positive_idx:
                    if coefs[idx] > 0:
                        st.markdown(f"✅ **{feature_names[idx]}** - Strong positive impact")
                
                # Top negative features
                top_negative_idx = np.argsort(coefs)[:3]
                
                st.markdown("\n**Factors that may require additional support:**")
                for idx in top_negative_idx:
                    if coefs[idx] < 0:
                        st.markdown(f"⚠️ **{feature_names[idx]}** - May indicate need for intervention")
    
    # Tab 5: Make Predictions
    with tab5:
        if not st.session_state.model_trained:
            st.info("👈 Please train models first from the Model Training tab")
        else:
            st.markdown("## 🎯 Individual Predictions")
            
            st.markdown("### Enter Participant Information")
            
            feature_names = st.session_state.feature_names
            
            # Create input form
            col1, col2, col3 = st.columns(3)
            
            input_data = {}
            
            for idx, feature in enumerate(feature_names):
                col_idx = idx % 3
                with [col1, col2, col3][col_idx]:
                    if 'Children' in feature:
                        input_data[feature] = st.number_input(
                            feature.replace('_', ' '),
                            min_value=0,
                            max_value=10,
                            value=0,
                            step=1
                        )
                    else:
                        input_data[feature] = st.selectbox(
                            feature.replace('_', ' '),
                            [0, 1],
                            format_func=lambda x: 'Yes' if x == 1 else 'No'
                        )
            
            if st.button("🔮 Predict Job Placement Probability", type="primary"):
                # Prepare input
                input_array = np.array([input_data[f] for f in feature_names]).reshape(1, -1)
                input_scaled = st.session_state.scaler.transform(input_array)
                
                # Get predictions from all models
                st.markdown("### 📊 Prediction Results")
                
                predictions = {}
                for name, model in st.session_state.models.items():
                    prob = model.predict_proba(input_scaled)[0][1]
                    predictions[name] = prob
                
                # Display gauge charts
                cols = st.columns(3)
                
                for idx, (name, prob) in enumerate(predictions.items()):
                    with cols[idx]:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': name},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightcoral"},
                                    {'range': [30, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Ensemble prediction
                ensemble_prob = np.mean(list(predictions.values()))
                
                st.markdown("### 🎯 Ensemble Prediction")
                
                if ensemble_prob >= threshold:
                    st.success(f"✅ **HIGH PROBABILITY** of job placement: {ensemble_prob:.1%}")
                else:
                    st.warning(f"⚠️ **LOW PROBABILITY** of job placement: {ensemble_prob:.1%}")
                
                st.markdown(f"**Recommendation:** {'Strong candidate for placement' if ensemble_prob >= threshold else 'May benefit from additional training/support'}")
                
                # Feature contribution (for Logistic Regression)
                st.markdown("### 📈 Feature Contribution Analysis")
                
                log_reg = st.session_state.models['Logistic Regression']
                coefs = log_reg.coef_[0]
                
                contributions = input_array[0] * coefs
                
                fig = go.Figure(go.Bar(
                    x=contributions,
                    y=feature_names,
                    orientation='h',
                    marker=dict(
                        color=contributions,
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title='How Each Feature Contributes to Prediction',
                    xaxis_title='Contribution',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
