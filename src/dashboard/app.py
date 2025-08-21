import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np
import time
from streamlit_shap import st_shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
import dice_ml
from dice_ml.utils import helpers

st.set_page_config(page_title="Explainable AI Dashboard", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("../models/models.joblib")

@st.cache_data
def load_sample():
    return pd.read_csv("../sample_inputs.csv")

@st.cache_resource
def create_explainers():
    """Create SHAP explainers for each model"""
    bundle = load_bundle()
    X_bg = load_sample()
    
    explainers = {}
    
    # For XGBoost: Use TreeExplainer with properly transformed data
    xgb_pipe = bundle["xgb"]
    X_bg_transformed = xgb_pipe.named_steps['prep'].transform(X_bg)
    
    # Convert sparse matrix to dense for TreeExplainer
    if hasattr(X_bg_transformed, 'toarray'):
        X_bg_dense = X_bg_transformed.toarray()
    else:
        X_bg_dense = X_bg_transformed
    
    # Use a smaller sample for efficiency
    sample_size = min(100, len(X_bg_dense))
    X_bg_sample = X_bg_dense[:sample_size]
    
    explainers["XGBoost"] = shap.TreeExplainer(xgb_pipe.named_steps['clf'], X_bg_sample)
    
    # For Logistic Regression: Use LinearExplainer with transformed data
    lr_pipe = bundle["lr"]
    lr_bg_transformed = lr_pipe.named_steps['prep'].transform(X_bg)
    
    # Convert sparse matrix to dense for LinearExplainer
    if hasattr(lr_bg_transformed, 'toarray'):
        lr_bg_dense = lr_bg_transformed.toarray()
    else:
        lr_bg_dense = lr_bg_transformed
    
    # Use a smaller sample for efficiency
    lr_sample_size = min(50, len(lr_bg_dense))
    lr_bg_sample = lr_bg_dense[:lr_sample_size]
    
    explainers["Logistic Regression"] = shap.LinearExplainer(lr_pipe.named_steps['clf'], lr_bg_sample)
    
    return explainers

def get_shap_values(model_name, model, X_data, explainers):
    """Get SHAP values for the selected model"""
    if model_name == "XGBoost":
        # Transform data for XGBoost
        X_transformed = model.named_steps['prep'].transform(X_data)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        return explainers[model_name].shap_values(X_transformed)
    else:
        # For Logistic Regression, also transform data first
        X_transformed = model.named_steps['prep'].transform(X_data)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        return explainers[model_name].shap_values(X_transformed)

bundle = load_bundle()
models = {"Logistic Regression": bundle["lr"], "XGBoost": bundle["xgb"]}
X_bg = load_sample()
feature_names = X_bg.columns.tolist()
explainers = create_explainers()

st.title("Explainable AI Dashboard for Black-Box Models")

with st.sidebar:
    model_name = st.selectbox("Model", list(models.keys()))
    model = models[model_name]
    idx = st.number_input("Row to explain (local)", min_value=0, max_value=len(X_bg)-1, value=0, step=1)
    show_shap_beeswarm = st.checkbox("Global: SHAP Beeswarm", value=False)
    show_shap_bar = st.checkbox("Global: SHAP Bar", value=False)
    show_local_waterfall = st.checkbox("Local: SHAP Waterfall", value=False)
    show_shap_narrative = st.checkbox("Local: SHAP Plain Language", value=False)
    show_lime_local = st.checkbox("Local: LIME", value=False)
    show_methods_analytics = st.checkbox("Methods Analytics", value=False)
    show_stability_audit = st.checkbox("Stability Audit (beta)", value=False)
    show_collinearity_check = st.checkbox("Collinearity Check", value=False)
    show_counterfactuals = st.checkbox("Counterfactuals (DiCE)", value=False)
    show_confidence_badge = st.checkbox("Confidence Badge", value=False)
    
    # Expert/Layperson mode
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Audience mode", ["Expert", "Layperson"])
    
    # Stability Audit parameters (only show when enabled)
    if show_stability_audit:
        st.sidebar.markdown("**Stability Parameters:**")
        runs = st.sidebar.slider("Perturbation runs", 3, 30, 5)
        noise = st.sidebar.slider("Numeric noise std (z-score units)", 0.0, 0.5, 0.1)

st.header("Data & Prediction")
st.dataframe(X_bg.head(10))
proba = model.predict_proba(X_bg)[:,1]
st.write(f"Selected row predicted probability: {proba[idx]:.3f}")

st.header("Global Explanations (Model-Level)")
# SHAP global - use our fixed explainer
try:
    with st.spinner("Computing SHAP values..."):
        shap_values = get_shap_values(model_name, model, X_bg, explainers)
        
        # Create SHAP Explanation object for plotting
        X_bg_transformed = model.named_steps['prep'].transform(X_bg)
        if hasattr(X_bg_transformed, 'toarray'):
            X_bg_transformed = X_bg_transformed.toarray()
        
        # Create the explanation object with the right base values
        if model_name == "XGBoost":
            base_values = explainers[model_name].expected_value
        else:
            base_values = explainers[model_name].expected_value
        
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=X_bg_transformed
        )

    if show_shap_beeswarm:
        st.subheader("SHAP Beeswarm (Global impact & direction)")
        st_shap(shap.plots.beeswarm(shap_explanation), height=350)

    if show_shap_bar:
        st.subheader("SHAP Bar (Mean absolute importance)")
        st_shap(shap.plots.bar(shap_explanation), height=350)

    st.header("Local Explanations (Instance-Level)")
    # SHAP waterfall for selected instance
    if show_local_waterfall:
        st.subheader("SHAP Waterfall")
        st_shap(shap.plots.waterfall(shap_explanation[idx]), height=350)
    
    if show_shap_narrative:
        # Add plain-language narrative for SHAP
        st.subheader("Plain-language explanation (SHAP)")
        try:
            # Get SHAP values for the selected instance
            shap_row = shap_explanation[idx]
            pred = model.predict_proba(X_bg.iloc[[idx]])[:,1][0]
            
            # Get base value (expected value)
            if model_name == "XGBoost":
                base = explainers[model_name].expected_value
            else:
                base = explainers[model_name].expected_value
            
            # Get feature names for the transformed data
            num_features = bundle['num']
            cat_features = bundle['cat']
            num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
            cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
            transformed_feature_names = num_feature_names + cat_feature_names
            
            # Create pairs of (feature_name, shap_value, original_raw_value)
            # For transformed features, we'll show the transformed feature name and original raw values where applicable
            pairs = []
            
            # Handle numerical features (direct mapping)
            for i, feat_name in enumerate(num_feature_names):
                original_feat = num_features[i]
                pairs.append((
                    original_feat,  # Use original feature name
                    shap_row.values[i],  # SHAP value
                    X_bg.iloc[idx][original_feat]  # Original raw value
                ))
            
            # Handle categorical features (encoded features mapped back to original)
            for i, feat_name in enumerate(cat_feature_names):
                shap_idx = len(num_feature_names) + i
                # Extract original feature name and value from encoded feature name
                if '_' in feat_name and '=' not in feat_name:
                    # Handle OneHotEncoder format like 'workclass_Private'
                    parts = feat_name.split('_', 1)
                    if len(parts) == 2:
                        original_feat, encoded_value = parts
                        # Only include if this encoded feature is "active" (value = 1 in one-hot encoding)
                        if shap_row.data[shap_idx] == 1.0:  # This encoded feature is active
                            pairs.append((
                                f"{original_feat}={encoded_value}",
                                shap_row.values[shap_idx],
                                encoded_value  # Show the categorical value
                            ))
                elif '=' in feat_name:
                    original_feat, encoded_value = feat_name.split('=', 1)
                    # Only include if this encoded feature is "active" (value = 1 in one-hot encoding)
                    if shap_row.data[shap_idx] == 1.0:  # This encoded feature is active
                        pairs.append((
                            f"{original_feat}={encoded_value}",
                            shap_row.values[shap_idx],
                            encoded_value  # Show the categorical value
                        ))
                else:
                    # Fallback for features without clear encoding pattern
                    if abs(shap_row.values[shap_idx]) > 0.001:  # Only include significant contributions
                        pairs.append((
                            feat_name,
                            shap_row.values[shap_idx],
                            shap_row.data[shap_idx]
                        ))
            
            # Sort by absolute contribution and take top 5
            pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:5]
            
            # Create narrative explanation
            st.write(f"**Model prediction: {pred:.1%} probability of earning >$50K**")
            st.write(f"**Baseline expectation: {base:.4f}** (average model output)")
            st.write("")
            st.write("**This prediction is influenced by these features (top 5 by impact):**")
            
            for name, contrib, raw in pairs_sorted:
                direction = "increases" if contrib > 0 else "decreases"
                impact_color = "🟢" if contrib > 0 else "🔴"
                
                # Format the contribution description
                if abs(contrib) > 0.001:
                    contrib_str = f"{abs(contrib):.3f}"
                else:
                    contrib_str = f"{abs(contrib):.1e}"
                
                st.write(f"{impact_color} **{name}** = `{raw}` {direction} the probability by {contrib_str}")
            
            # Add explanatory note
            st.info("""
            **How to read this:**
            - 🟢 Green features push the prediction toward higher probability (>$50K)
            - 🔴 Red features push the prediction toward lower probability (≤$50K)
            - Numbers show the magnitude of each feature's impact
            - All contributions sum up to explain the difference from the baseline
            """)
            
        except Exception as e:
            st.error(f"Error generating plain-language explanation: {str(e)}")
            st.write("SHAP values are available in the waterfall plot above.")
    
    if show_lime_local:
        st.subheader("LIME Local Explanation")
        try:
            # For LIME, we need to work with preprocessed numerical data
            # Transform the background data first
            X_bg_transformed = model.named_steps['prep'].transform(X_bg)
            if hasattr(X_bg_transformed, 'toarray'):
                X_bg_dense = X_bg_transformed.toarray()
            else:
                X_bg_dense = X_bg_transformed
            
            # Get feature names for the transformed data
            # Get feature names from the preprocessing pipeline
            num_features = bundle['num']
            cat_features = bundle['cat']
            
            # Create feature names for transformed data
            num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
            cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
            transformed_feature_names = num_feature_names + cat_feature_names
            
            # Create LIME explainer with transformed numerical data
            explainer_lime = LimeTabularExplainer(
                training_data=X_bg_dense,
                feature_names=transformed_feature_names,
                class_names=["<=50K", ">50K"],
                discretize_continuous=True,
                mode="classification"
            )
            
            # Create prediction function that works with transformed data
            def predict_proba_fn(X_array):
                """Prediction function for LIME that handles transformed data"""
                try:
                    # X_array is already transformed numerical data
                    if X_array.ndim == 1:
                        X_array = X_array.reshape(1, -1)
                    return model.named_steps['clf'].predict_proba(X_array)
                except Exception as e:
                    st.error(f"Error in LIME prediction function: {str(e)}")
                    # Return dummy predictions to prevent LIME from crashing
                    return np.array([[0.5, 0.5]] * len(X_array))
            
            # Transform the selected instance
            selected_instance = X_bg.iloc[[idx]]  # Keep as DataFrame
            selected_transformed = model.named_steps['prep'].transform(selected_instance)
            if hasattr(selected_transformed, 'toarray'):
                selected_dense = selected_transformed.toarray()[0]
            else:
                selected_dense = selected_transformed[0]
            
            # Get explanation for the selected instance
            with st.spinner("Computing LIME explanation..."):
                lime_exp = explainer_lime.explain_instance(
                    data_row=selected_dense,
                    predict_fn=predict_proba_fn,
                    num_features=10,
                    top_labels=1
                )
            
            # Display LIME results
            weights = lime_exp.as_list(label=lime_exp.top_labels[0])
            st.write("**Top local features (LIME):**")
            
            # Create a more visual display
            col1, col2 = st.columns([3, 1])
            with col1:
                for i, (feature, weight) in enumerate(weights):
                    color = "🟢" if weight > 0 else "🔴"
                    st.write(f"{color} **{feature}**: {weight:.4f}")
            
            with col2:
                st.write("**Legend:**")
                st.write("🟢 Increases probability")
                st.write("🔴 Decreases probability")
            
            # Show the original instance being explained
            st.write("**Original instance being explained:**")
            instance_data = X_bg.iloc[idx].to_dict()
            cols = st.columns(3)
            for i, (key, value) in enumerate(instance_data.items()):
                with cols[i % 3]:
                    st.metric(key, value)
                    
        except Exception as e:
            st.error(f"Error computing LIME explanation: {str(e)}")
            st.info("LIME explanations work best with numerical data. The error might be due to categorical data handling.")
            # Show some debug info
            st.write("Debug info:")
            st.write(f"Selected instance shape: {X_bg.iloc[idx].shape}")
            st.write(f"Feature names: {feature_names}")
            st.write(f"Data types: {X_bg.dtypes.to_dict()}")

    # Comparison section if both SHAP and LIME are enabled
    if show_local_waterfall and show_lime_local:
        st.header("SHAP vs LIME Comparison")
        st.write("""
        **SHAP (SHapley Additive exPlanations)**:
        - Based on game theory (Shapley values)
        - Guarantees that explanations sum to difference from baseline
        - Model-specific optimizations available (TreeExplainer, LinearExplainer)
        - More computationally intensive but theoretically grounded
        
        **LIME (Local Interpretable Model-agnostic Explanations)**:
        - Fits local linear model around the instance
        - Model-agnostic (works with any black box)
        - Faster computation through sampling
        - Approximate explanations based on local perturbations
        
        Both methods provide complementary insights into model predictions!
        """)
    
    # Methods Analytics section
    if show_methods_analytics and (show_local_waterfall or show_lime_local):
        st.header("Methods Analytics")
        
        try:
            analytics_results = {}
            
            # Initialize variables for timing
            t_shap = 0
            t_lime = 0
            shap_local = None
            lime_local = None
            
            # Time SHAP local computation (if enabled)
            if show_local_waterfall:
                st.write("**Computing SHAP timing and features...**")
                t0 = time.time()
                
                # Extract SHAP values for current instance
                sv = shap_explanation[idx]
                
                # Get feature names for proper mapping
                num_features = bundle['num']
                cat_features = bundle['cat']
                num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
                cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
                all_feature_names = num_feature_names + cat_feature_names
                
                # Create series with proper feature names
                shap_local = pd.Series(sv.values, index=all_feature_names)
                shap_top = shap_local.abs().sort_values(ascending=False).head(10)
                
                t_shap = (time.time() - t0) * 1000  # Convert to milliseconds
                analytics_results['shap_timing'] = t_shap
                analytics_results['shap_features'] = shap_top
                
            # Time LIME local computation (if enabled)
            if show_lime_local:
                st.write("**Computing LIME timing and features...**")
                t1 = time.time()
                
                # Recompute LIME for timing purposes
                # Transform the background data
                X_bg_transformed = model.named_steps['prep'].transform(X_bg)
                if hasattr(X_bg_transformed, 'toarray'):
                    X_bg_dense = X_bg_transformed.toarray()
                else:
                    X_bg_dense = X_bg_transformed
                
                # Get feature names
                num_features = bundle['num']
                cat_features = bundle['cat']
                num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
                cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
                transformed_feature_names = num_feature_names + cat_feature_names
                
                # Create LIME explainer
                explainer_lime = LimeTabularExplainer(
                    training_data=X_bg_dense,
                    feature_names=transformed_feature_names,
                    class_names=["<=50K", ">50K"],
                    discretize_continuous=True,
                    mode="classification"
                )
                
                # Prediction function
                def predict_proba_fn(X_array):
                    if X_array.ndim == 1:
                        X_array = X_array.reshape(1, -1)
                    return model.named_steps['clf'].predict_proba(X_array)
                
                # Transform the selected instance
                selected_instance = X_bg.iloc[[idx]]
                selected_transformed = model.named_steps['prep'].transform(selected_instance)
                if hasattr(selected_transformed, 'toarray'):
                    selected_dense = selected_transformed.toarray()[0]
                else:
                    selected_dense = selected_transformed[0]
                
                # Get LIME explanation
                lime_exp = explainer_lime.explain_instance(
                    data_row=selected_dense,
                    predict_fn=predict_proba_fn,
                    num_features=10,
                    top_labels=1
                )
                
                # Extract LIME results
                lime_pairs = lime_exp.as_list(label=lime_exp.top_labels[0])
                lime_local = pd.Series({k: v for k, v in lime_pairs})
                
                t_lime = (time.time() - t1) * 1000  # Convert to milliseconds
                analytics_results['lime_timing'] = t_lime
                analytics_results['lime_features'] = lime_local
                
            # Display timing results
            st.subheader("Performance Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'shap_timing' in analytics_results:
                    st.metric("SHAP Latency", f"{analytics_results['shap_timing']:.1f} ms")
                else:
                    st.write("SHAP not computed (not enabled)")
                    
            with col2:
                if 'lime_timing' in analytics_results:
                    st.metric("LIME Latency", f"{analytics_results['lime_timing']:.1f} ms")
                else:
                    st.write("LIME not computed (not enabled)")
            
            # Feature agreement analysis (if both are available)
            if shap_local is not None and lime_local is not None:
                st.subheader("Feature Agreement Analysis")
                
                # Find common features by trying to map them
                # This is complex due to different feature naming conventions
                shap_features = set(shap_local.index)
                lime_features = set(lime_local.index)
                
                # For better comparison, let's look at the top features from each
                shap_top_names = shap_top.head(5).index.tolist()
                lime_top_names = lime_local.abs().sort_values(ascending=False).head(5).index.tolist()
                
                # Simple overlap count
                # Try to find partial matches for categorical features
                matches = 0
                shap_simplified = []
                lime_simplified = []
                
                for shap_feat in shap_top_names:
                    shap_base = shap_feat.split('_')[0] if '_' in shap_feat else shap_feat
                    shap_simplified.append(shap_base)
                    
                for lime_feat in lime_top_names:
                    lime_base = lime_feat.split('_')[0] if '_' in lime_feat else lime_feat
                    lime_simplified.append(lime_base)
                    
                # Count overlapping base feature names
                shap_set = set(shap_simplified)
                lime_set = set(lime_simplified)
                common_base_features = shap_set.intersection(lime_set)
                
                overlap_ratio = len(common_base_features) / max(len(shap_set), len(lime_set)) if shap_set or lime_set else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Feature Overlap", f"{len(common_base_features)}/{min(5, len(shap_set), len(lime_set))}")
                with col2:
                    st.metric("Overlap Ratio", f"{overlap_ratio:.1%}")
                with col3:
                    if overlap_ratio >= 0.6:
                        agreement_level = "High 🟢"
                    elif overlap_ratio >= 0.3:
                        agreement_level = "Medium 🟡"
                    else:
                        agreement_level = "Low 🔴"
                    st.metric("Agreement", agreement_level)
                
                # Show top features side by side
                st.write("**Top 5 Features Comparison:**")
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.write("**SHAP Top Features:**")
                    for i, (feat, val) in enumerate(shap_top.head(5).items(), 1):
                        direction = "🟢 +" if shap_local[feat] > 0 else "🔴 -"
                        st.write(f"{i}. {direction} {feat}: {abs(val):.3f}")
                
                with comp_col2:
                    st.write("**LIME Top Features:**")
                    lime_top_sorted = lime_local.abs().sort_values(ascending=False).head(5)
                    for i, (feat, val) in enumerate(lime_top_sorted.items(), 1):
                        direction = "🟢 +" if lime_local[feat] > 0 else "🔴 -"
                        st.write(f"{i}. {direction} {feat}: {abs(val):.3f}")
                
                # Summary insights
                st.info(f"""
                **Summary:**
                - **Timing**: {'SHAP' if t_shap < t_lime else 'LIME'} was faster by {abs(t_shap - t_lime):.1f}ms
                - **Agreement**: {len(common_base_features)} of top 5 base features overlap between methods
                - **Consistency**: {'High' if overlap_ratio >= 0.6 else 'Medium' if overlap_ratio >= 0.3 else 'Low'} agreement suggests {'consistent' if overlap_ratio >= 0.6 else 'partially consistent' if overlap_ratio >= 0.3 else 'divergent'} explanations
                """)
                
                # Store variables for confidence badge
                st.session_state['agreement_score'] = overlap_ratio
                st.session_state['performance_score'] = 1.0 - min(0.5, (t_shap + t_lime) / 1000.0)  # Normalize latency
            
            else:
                st.info("Enable both SHAP Waterfall and LIME to see feature agreement analysis.")
                
        except Exception as e:
            st.error(f"Error in methods analytics: {str(e)}")
            st.write("Make sure both SHAP and LIME explanations are enabled for full analytics.")

    # Collinearity Check section
    if show_collinearity_check:
        st.header("Collinearity Check")
        st.write("**Detect strongly correlated features that may affect attribution reliability**")
        
        try:
            # Compute correlation matrix for numeric features only
            numeric_data = X_bg.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr = numeric_data.corr().abs()
                
                # Get upper triangle of correlation matrix
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                
                # Find highly correlated pairs (>0.8)
                high_pairs = []
                for i in range(upper.shape[0]):
                    for j in range(upper.shape[1]):
                        if pd.notna(upper.iloc[i, j]) and upper.iloc[i, j] > 0.8:
                            high_pairs.append((corr.columns[i], corr.columns[j], upper.iloc[i, j]))
                
                if high_pairs:
                    st.warning(f"⚠️ High collinearity detected among numeric features (>|0.8|). This can affect attribution reliability.")
                    st.write("**Highly correlated feature pairs:**")
                    
                    # Display top 10 pairs
                    for i, (feature_a, feature_b, correlation) in enumerate(high_pairs[:10]):
                        st.write(f"• **{feature_a}** ~ **{feature_b}**: {correlation:.3f}")
                    
                    if len(high_pairs) > 10:
                        st.write(f"... and {len(high_pairs) - 10} more pairs")
                    
                    # Provide guidance
                    st.info("""
                    **Impact on explanations:**
                    - Highly correlated features may show unstable attribution
                    - SHAP/LIME may distribute importance unpredictably between correlated features
                    - Consider feature selection or grouping for more reliable explanations
                    """)
                    
                    # Show correlation heatmap for highly correlated features
                    if len(high_pairs) > 0:
                        st.subheader("Correlation Heatmap (High Correlation Features)")
                        
                        # Get unique features involved in high correlations
                        involved_features = set()
                        for feature_a, feature_b, _ in high_pairs:
                            involved_features.add(feature_a)
                            involved_features.add(feature_b)
                        
                        involved_features = list(involved_features)[:10]  # Limit for display
                        
                        if len(involved_features) > 1:
                            subset_corr = corr.loc[involved_features, involved_features]
                            
                            # Create a simple text-based heatmap
                            st.write("**Correlation Matrix (subset):**")
                            
                            # Display correlation matrix
                            corr_display = subset_corr.round(3)
                            st.dataframe(corr_display, use_container_width=True)
                else:
                    st.success("✅ No high collinearity detected (all correlations ≤0.8)")
                    st.write("Feature attributions should be relatively stable.")
                    
                # Show correlation summary
                st.subheader("Correlation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_corr = upper.max().max()
                    st.metric("Max Correlation", f"{max_corr:.3f}" if not pd.isna(max_corr) else "N/A")
                
                with col2:
                    mean_corr = upper.mean().mean()
                    st.metric("Mean Correlation", f"{mean_corr:.3f}" if not pd.isna(mean_corr) else "N/A")
                
                with col3:
                    high_corr_count = len(high_pairs)
                    st.metric("High Corr. Pairs", high_corr_count)
                    
            else:
                st.info("Not enough numeric features for correlation analysis (need at least 2).")
                
        except Exception as e:
            st.error(f"Error in collinearity check: {str(e)}")
            st.write("Make sure there are numeric features in the dataset.")

    # Counterfactuals section
    if show_counterfactuals:
        st.header("Counterfactuals (DiCE)")
        st.write("**Generate actionable 'what-if' examples to flip the prediction**")
        
        try:
            # Current prediction
            current_pred = model.predict_proba(X_bg.iloc[[idx]])[:,1][0]
            current_class = ">50K" if current_pred >= 0.5 else "<=50K"
            
            # User can select desired outcome
            desired = st.selectbox("Desired class", [">50K", "<=50K"], 
                                 index=0 if current_class == "<=50K" else 1)
            
            if st.button("Generate Counterfactuals"):
                with st.spinner("Generating counterfactual examples..."):
                    try:
                        # Determine desired class for DiCE
                        if desired != current_class:
                            desired_class = 1 if desired == ">50K" else 0
                        else:
                            st.info(f"Current prediction is already {current_class}. Showing counterfactuals for opposite class.")
                            desired_class = 0 if current_class == ">50K" else 1
                        
                        # Create DiCE data and model objects
                        continuous_features = X_bg.select_dtypes(include=[np.number]).columns.tolist()
                        
                        # Create a simple dataset for DiCE (it needs target column)
                        X_bg_with_target = X_bg.copy()
                        X_bg_with_target['income'] = (model.predict_proba(X_bg)[:,1] >= 0.5).astype(int)
                        
                        # Initialize DiCE
                        d = dice_ml.Data(dataframe=X_bg_with_target, 
                                       continuous_features=continuous_features,
                                       outcome_name='income')
                        
                        # Create a wrapper for the model
                        class ModelWrapper:
                            def __init__(self, model):
                                self.model = model
                                
                            def predict_proba(self, X):
                                # Remove the target column if present
                                if 'income' in X.columns:
                                    X = X.drop('income', axis=1)
                                return self.model.predict_proba(X)
                            
                            def predict(self, X):
                                if 'income' in X.columns:
                                    X = X.drop('income', axis=1)
                                return (self.model.predict_proba(X)[:,1] >= 0.5).astype(int)
                        
                        model_wrapper = ModelWrapper(model)
                        m = dice_ml.Model(model=model_wrapper, backend="sklearn")
                        
                        # Generate counterfactuals
                        exp = dice_ml.Dice(d, m, method="random")
                        
                        # Prepare query instance (without target column)
                        query_instance = X_bg.iloc[[idx]]  # Original features only, no target
                        
                        # Generate counterfactuals
                        dice_exp = exp.generate_counterfactuals(
                            query_instance, 
                            total_CFs=3, 
                            desired_class=desired_class
                        )
                        
                        # Display results
                        if dice_exp.cf_examples_list[0].final_cfs_df is not None:
                            st.success("✅ Counterfactual examples generated successfully!")
                            
                            # Show original instance
                            st.subheader("Original Instance")
                            orig_data = X_bg.iloc[idx].to_dict()
                            orig_cols = st.columns(3)
                            for i, (key, value) in enumerate(list(orig_data.items())[:6]):  # Show first 6 features
                                with orig_cols[i % 3]:
                                    st.metric(key, value)
                            
                            st.write(f"**Current prediction**: {current_class} (probability: {current_pred:.3f})")
                            
                            # Show counterfactuals
                            st.subheader("Suggested Counterfactual Changes")
                            
                            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                            if 'income' in cf_df.columns:
                                cf_df = cf_df.drop('income', axis=1)
                            
                            # Show only the differences
                            original_row = X_bg.iloc[idx]
                            
                            for cf_idx, (_, cf_row) in enumerate(cf_df.iterrows()):
                                st.write(f"**Counterfactual {cf_idx + 1}:**")
                                
                                changes = []
                                for feature in X_bg.columns:
                                    if feature in cf_row.index:
                                        original_val = original_row[feature]
                                        cf_val = cf_row[feature]
                                        
                                        # Check if values are different (handle both numeric and categorical)
                                        if pd.isna(original_val) and pd.isna(cf_val):
                                            continue
                                        elif pd.isna(original_val) or pd.isna(cf_val):
                                            changes.append(f"• **{feature}**: {original_val} → {cf_val}")
                                        elif isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                                            if abs(original_val - cf_val) > 1e-6:
                                                changes.append(f"• **{feature}**: {original_val} → {cf_val}")
                                        elif str(original_val) != str(cf_val):
                                            changes.append(f"• **{feature}**: {original_val} → {cf_val}")
                                
                                if changes:
                                    for change in changes[:5]:  # Show top 5 changes
                                        st.write(change)
                                else:
                                    st.write("• No significant changes detected")
                                
                                # Predict the counterfactual outcome
                                cf_pred = model.predict_proba(cf_row.to_frame().T)[:,1][0]
                                cf_class = ">50K" if cf_pred >= 0.5 else "<=50K"
                                st.write(f"  **Predicted outcome**: {cf_class} (probability: {cf_pred:.3f})")
                                st.write("")
                        else:
                            st.warning("⚠️ Could not generate valid counterfactuals. Try adjusting the model or data.")
                            
                    except Exception as e:
                        st.error(f"Error generating counterfactuals: {str(e)}")
                        st.write("This might be due to model compatibility or data format issues.")
                        
                        # Show some debug info
                        st.write("**Debug info:**")
                        st.write(f"Current prediction: {current_pred:.3f}")
                        st.write(f"Desired class: {desired}")
                        st.write(f"Continuous features: {continuous_features}")
            
        except Exception as e:
            st.error(f"Error in counterfactuals setup: {str(e)}")
            st.write("Make sure DiCE is properly installed and the model is compatible.")

    # Confidence Badge section
    if show_confidence_badge:
        st.header("Explanation Confidence Badge")
        st.write("**Combined assessment of explanation reliability and performance**")
        
        try:
            # Initialize score
            score = 0.0
            components = {}
            
            # Agreement component (from methods analytics if available)
            if 'agreement_score' in st.session_state:
                agreement_score = st.session_state['agreement_score']
                components['Agreement'] = agreement_score
                score += agreement_score * 0.4
            else:
                components['Agreement'] = "N/A (enable Methods Analytics)"
            
            # Stability component (from stability audit if available)
            if show_stability_audit and 'stability_audit_run' in st.session_state and st.session_state['stability_audit_run']:
                if 'stability_score' in st.session_state:
                    stability_score = st.session_state['stability_score']
                    components['Stability'] = stability_score
                    score += stability_score * 0.3
                else:
                    components['Stability'] = "Audit completed, no score available"
            elif show_stability_audit:
                components['Stability'] = "Audit enabled, click 'Run Stability Audit'"
            else:
                components['Stability'] = "N/A (enable Stability Audit)"
            
            # Performance component (latency penalty)
            if 'performance_score' in st.session_state:
                performance_score = st.session_state['performance_score']
                components['Performance'] = performance_score
                score += performance_score * 0.1
            else:
                components['Performance'] = "N/A (enable Methods Analytics)"
            
            # Collinearity component
            if show_collinearity_check:
                try:
                    # Check for high correlations
                    numeric_data = X_bg.select_dtypes(include=[np.number])
                    if len(numeric_data.columns) > 1:
                        corr = numeric_data.corr().abs()
                        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        max_corr = upper.max().max()
                        
                        if pd.isna(max_corr):
                            collinearity_score = 1.0
                        elif max_corr > 0.9:
                            collinearity_score = 0.3  # Very high correlation
                        elif max_corr > 0.8:
                            collinearity_score = 0.6  # High correlation
                        else:
                            collinearity_score = 1.0  # Acceptable correlation
                            
                        components['Collinearity'] = collinearity_score
                        score += collinearity_score * 0.2
                    else:
                        components['Collinearity'] = 1.0
                        score += 0.2
                except:
                    components['Collinearity'] = "N/A"
            else:
                components['Collinearity'] = "N/A (enable Collinearity Check)"
            
            # Normalize score
            max_possible_score = 0.4 + 0.3 + 0.1 + 0.2  # Agreement + Stability + Performance + Collinearity
            actual_components = len([v for v in components.values() if isinstance(v, (int, float))])
            
            if actual_components > 0 and score > 0:
                # Normalize based on available components
                available_weight = 0.0
                if isinstance(components['Agreement'], (int, float)):
                    available_weight += 0.4
                if isinstance(components['Stability'], (int, float)):
                    available_weight += 0.3
                if isinstance(components['Performance'], (int, float)):
                    available_weight += 0.1
                if isinstance(components['Collinearity'], (int, float)):
                    available_weight += 0.2
                
                if available_weight > 0:
                    score = min(1.0, score / available_weight)
                else:
                    score = 0.0
            else:
                score = 0.0
            
            # Determine confidence level
            if score >= 0.7:
                confidence_level = "High"
                confidence_color = "🟢"
                confidence_message = "Explanations are highly reliable"
            elif score >= 0.4:
                confidence_level = "Medium"
                confidence_color = "🟡"
                confidence_message = "Explanations are moderately reliable"
            else:
                confidence_level = "Low"
                confidence_color = "🔴"
                confidence_message = "Exercise caution with explanations"
            
            # Display confidence badge
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid #ddd; border-radius: 10px; background-color: #f9f9f9;">
                    <h2>{confidence_color} {confidence_level} Confidence</h2>
                    <p style="font-size: 18px;"><strong>Score: {score:.2f}/1.0</strong></p>
                    <p style="font-style: italic;">{confidence_message}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show component breakdown
            st.subheader("Component Breakdown")
            
            # Filter numeric components for column layout
            numeric_components = {k: v for k, v in components.items() if isinstance(v, (int, float))}
            non_numeric_components = {k: v for k, v in components.items() if not isinstance(v, (int, float))}
            
            if numeric_components:
                comp_cols = st.columns(len(numeric_components))
                for idx, (component, value) in enumerate(numeric_components.items()):
                    with comp_cols[idx]:
                        st.metric(component, f"{value:.2f}")
            
            # Show non-numeric components separately
            if non_numeric_components:
                st.write("**Component Status:**")
                for component, status in non_numeric_components.items():
                    if "N/A" in str(status):
                        st.write(f"• **{component}**: {status}")
                    elif "click" in str(status).lower():
                        st.write(f"• **{component}**: ⏳ {status}")
                    else:
                        st.write(f"• **{component}**: ℹ️ {status}")
            
            # Show actionable guidance for improving confidence
            missing_or_pending = [k for k, v in components.items() if not isinstance(v, (int, float))]
            if missing_or_pending:
                guidance_messages = []
                for component in missing_or_pending:
                    status = components[component]
                    if "enable" in status.lower():
                        guidance_messages.append(f"Enable {component}")
                    elif "click" in status.lower():
                        guidance_messages.append(f"Run {component} analysis")
                    elif "analytics" in status.lower():
                        guidance_messages.append(f"Enable Methods Analytics with both SHAP and LIME")
                
                if guidance_messages:
                    st.info(f"**To improve confidence assessment**: {', '.join(guidance_messages)}")
            
            # Interpretation guide
            st.subheader("Interpretation Guide")
            st.write("""
            **Confidence Levels:**
            - 🟢 **High (≥0.7)**: Explanations are reliable and consistent
            - 🟡 **Medium (≥0.4)**: Explanations are moderately reliable, consider cross-validation
            - 🔴 **Low (<0.4)**: Explanations may be unreliable, use with caution
            
            **Components:**
            - **Agreement**: How well SHAP and LIME agree on feature importance
            - **Stability**: How consistent explanations are under small perturbations  
            - **Performance**: Speed of explanation generation (faster = better)
            - **Collinearity**: Impact of correlated features on attribution reliability
            """)
            
        except Exception as e:
            st.error(f"Error computing confidence badge: {str(e)}")
            st.write("Enable other features (Methods Analytics, Stability Audit, etc.) for full confidence assessment.")

    # Expert/Layperson narrative (integrate with existing plain language section)
    if mode == "Layperson" and show_shap_narrative:
        st.header("Top Actionable Levers")
        st.write("**Simple, actionable insights for changing your outcome**")
        
        try:
            # Use the pairs from the SHAP narrative section
            if 'pairs_sorted' in locals():
                top_changes = pairs_sorted[:3]  # Top 3 most impactful features
                
                current_pred = model.predict_proba(X_bg.iloc[[idx]])[:,1][0]
                current_class = ">50K" if current_pred >= 0.5 else "<=50K"
                
                st.write(f"**Current prediction**: {current_class}")
                st.write("**To change your outcome, consider adjusting:**")
                
                for i, (name, contrib, raw) in enumerate(top_changes, 1):
                    if contrib > 0 and current_class == "<=50K":
                        direction = "increase"
                        action = f"Increasing {name} would help reach >$50K income"
                    elif contrib < 0 and current_class == ">50K":
                        direction = "decrease" 
                        action = f"Decreasing {name} would help maintain >$50K income"
                    elif contrib > 0 and current_class == ">50K":
                        direction = "maintain/increase"
                        action = f"Maintaining or increasing {name} helps keep >$50K income"
                    else:
                        direction = "address"
                        action = f"Addressing {name} could improve your outcome"
                    
                    st.write(f"{i}. **{action}**")
                    st.write(f"   Current value: `{raw}`, Impact: {abs(contrib):.3f}")
                
                if show_counterfactuals:
                    st.info("💡 **Tip**: See the Counterfactuals section above for specific target values to achieve your desired outcome.")
            else:
                st.info("Enable 'Local: SHAP Plain Language' to see actionable insights.")
                
        except Exception as e:
            st.error(f"Error generating actionable insights: {str(e)}")

    # Stability Audit section
    if show_stability_audit:
        st.header("Stability Audit (beta)")
        st.write("**Test explanation robustness by adding small perturbations to input data**")
        
        # Initialize stability state if not present
        if 'stability_audit_run' not in st.session_state:
            st.session_state['stability_audit_run'] = False
        
        if st.button("Run Stability Audit"):
            # Mark that audit has been run
            st.session_state['stability_audit_run'] = True
            
            try:
                # Get parameters
                runs = runs if 'runs' in locals() else 5
                noise = noise if 'noise' in locals() else 0.1
                
                shap_ranks, lime_ranks = [], []
                base_row = X_bg.iloc[idx].copy()  # This returns a Series  
                numeric_cols = X_bg.select_dtypes(include=[np.number]).columns.tolist()  # ✅ Fixed
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for run_idx in range(runs):
                    status_text.text(f"Running perturbation {run_idx + 1}/{runs}...")
                    progress_bar.progress((run_idx + 1) / runs)
                    
                    # Create perturbed row
                    row = base_row.copy()
                    if numeric_cols and noise > 0:
                        # Add noise to numeric features (standardized by feature std)
                        for col in numeric_cols:
                            col_std = X_bg[col].std()
                            if col_std > 0:  # Avoid division by zero
                                noise_value = np.random.normal(0, noise * col_std)
                                row[col] = row[col] + noise_value
                    
                    row_df = pd.DataFrame([row], columns=X_bg.columns)
                    
                    # SHAP local explanation for perturbed instance
                    if show_local_waterfall or show_methods_analytics:
                        try:
                            sv_perturbed = get_shap_values(model_name, model, row_df, explainers)
                            
                            # Get feature names for proper mapping
                            num_features = bundle['num']
                            cat_features = bundle['cat']
                            num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
                            cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
                            all_feature_names = num_feature_names + cat_feature_names
                            
                            # Create series with proper feature names
                            shap_local_perturbed = pd.Series(sv_perturbed[0], index=all_feature_names)
                            shap_top_perturbed = shap_local_perturbed.abs().sort_values(ascending=False)
                            shap_ranks.append(shap_top_perturbed.index.tolist()[:10])  # Top 10 features
                            
                        except Exception as e:
                            st.warning(f"SHAP computation failed for run {run_idx + 1}: {str(e)}")
                            continue
                    
                    # LIME local explanation for perturbed instance
                    if show_lime_local or show_methods_analytics:
                        try:
                            # Transform the perturbed instance
                            row_transformed = model.named_steps['prep'].transform(row_df)
                            if hasattr(row_transformed, 'toarray'):
                                row_dense = row_transformed.toarray()[0]
                            else:
                                row_dense = row_transformed[0]
                            
                            # Get feature names for the transformed data
                            num_features = bundle['num']
                            cat_features = bundle['cat']
                            num_feature_names = model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_features).tolist()
                            cat_feature_names = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
                            transformed_feature_names = num_feature_names + cat_feature_names
                            
                            # Create LIME explainer (using cached from earlier if available)
                            X_bg_transformed = model.named_steps['prep'].transform(X_bg)
                            if hasattr(X_bg_transformed, 'toarray'):
                                X_bg_dense = X_bg_transformed.toarray()
                            else:
                                X_bg_dense = X_bg_transformed
                            
                            explainer_lime = LimeTabularExplainer(
                                training_data=X_bg_dense,
                                feature_names=transformed_feature_names,
                                class_names=["<=50K", ">50K"],
                                discretize_continuous=True,
                                mode="classification"
                            )
                            
                            def predict_proba_fn(X_array):
                                if X_array.ndim == 1:
                                    X_array = X_array.reshape(1, -1)
                                return model.named_steps['clf'].predict_proba(X_array)
                            
                            # Get LIME explanation for perturbed instance
                            lime_exp = explainer_lime.explain_instance(
                                data_row=row_dense,
                                predict_fn=predict_proba_fn,
                                num_features=10,
                                top_labels=1
                            )
                            
                            # Extract LIME results
                            lime_pairs = lime_exp.as_list(label=lime_exp.top_labels[0])
                            lime_series = pd.Series({k: abs(v) for k, v in lime_pairs})
                            lime_top_features = lime_series.sort_values(ascending=False).index.tolist()
                            lime_ranks.append(lime_top_features[:10])  # Top 10 features
                            
                        except Exception as e:
                            st.warning(f"LIME computation failed for run {run_idx + 1}: {str(e)}")
                            continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Calculate stability metrics
                def topk_overlap(ranks_list, k=10):
                    """Calculate average overlap of top-k features across runs"""
                    if not ranks_list or len(ranks_list) < 2:
                        return 0.0
                    
                    overlaps = []
                    base_set = set(ranks_list[0][:k])
                    
                    for ranking in ranks_list[1:]:
                        comparison_set = set(ranking[:k])
                        overlap = len(base_set.intersection(comparison_set)) / k
                        overlaps.append(overlap)
                    
                    return np.mean(overlaps) if overlaps else 0.0
                
                # Display results
                st.subheader("Stability Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Perturbation Runs", runs)
                    
                with col2:
                    if shap_ranks:
                        shap_stability = topk_overlap(shap_ranks, k=10)
                        stability_color = "🟢" if shap_stability >= 0.8 else "🟡" if shap_stability >= 0.6 else "🔴"
                        st.metric("SHAP Stability", f"{shap_stability*100:.1f}%", delta=stability_color)
                        # Store SHAP stability immediately
                        st.session_state['shap_stability_score'] = shap_stability
                    else:
                        st.metric("SHAP Stability", "N/A")
                        
                with col3:
                    if lime_ranks:
                        lime_stability = topk_overlap(lime_ranks, k=10)
                        stability_color = "🟢" if lime_stability >= 0.8 else "🟡" if lime_stability >= 0.6 else "🔴"
                        st.metric("LIME Stability", f"{lime_stability*100:.1f}%", delta=stability_color)
                        # Store LIME stability immediately
                        st.session_state['lime_stability_score'] = lime_stability
                    else:
                        st.metric("LIME Stability", "N/A")
                
                # Detailed analysis
                if shap_ranks or lime_ranks:
                    st.subheader("Detailed Analysis")
                    
                    if shap_ranks and lime_ranks:
                        comp_col1, comp_col2 = st.columns(2)
                        
                        with comp_col1:
                            st.write("**SHAP Feature Consistency:**")
                            # Show which features appear most consistently
                            all_shap_features = [feat for ranking in shap_ranks for feat in ranking[:5]]
                            shap_freq = pd.Series(all_shap_features).value_counts()
                            for feat, count in shap_freq.head(5).items():
                                consistency = count / len(shap_ranks) * 100
                                st.write(f"• {feat}: {consistency:.0f}% of runs")
                        
                        with comp_col2:
                            st.write("**LIME Feature Consistency:**")
                            all_lime_features = [feat for ranking in lime_ranks for feat in ranking[:5]]
                            lime_freq = pd.Series(all_lime_features).value_counts()
                            for feat, count in lime_freq.head(5).items():
                                consistency = count / len(lime_ranks) * 100
                                st.write(f"• {feat}: {consistency:.0f}% of runs")
                    
                    # Interpretation guide
                    st.info(f"""
                    **Stability Interpretation:**
                    - 🟢 **High (≥80%)**: Explanations are very consistent across perturbations
                    - 🟡 **Medium (≥60%)**: Explanations show moderate variability  
                    - 🔴 **Low (<60%)**: Explanations are sensitive to input changes
                    
                    **Parameters used**: {runs} runs, {noise:.2f} std noise on numeric features
                    """)
                    
                    # Recommendations based on results
                    if shap_ranks and lime_ranks:
                        avg_stability = (shap_stability + lime_stability) / 2
                        if avg_stability >= 0.8:
                            st.success("✅ **High stability**: Explanations are reliable for this instance")
                        elif avg_stability >= 0.6:
                            st.warning("⚠️ **Moderate stability**: Consider averaging multiple explanations")
                        else:
                            st.error("❌ **Low stability**: Exercise caution when interpreting explanations")
                        
                        # Store combined stability for confidence badge
                        st.session_state['stability_score'] = avg_stability
                    elif shap_ranks:
                        # Only SHAP available
                        st.session_state['stability_score'] = shap_stability
                        if shap_stability >= 0.8:
                            st.success("✅ **High SHAP stability**: SHAP explanations are reliable for this instance")
                        elif shap_stability >= 0.6:
                            st.warning("⚠️ **Moderate SHAP stability**: Consider running multiple explanations")
                        else:
                            st.error("❌ **Low SHAP stability**: Exercise caution with SHAP explanations")
                    elif lime_ranks:
                        # Only LIME available
                        st.session_state['stability_score'] = lime_stability
                        if lime_stability >= 0.8:
                            st.success("✅ **High LIME stability**: LIME explanations are reliable for this instance")
                        elif lime_stability >= 0.6:
                            st.warning("⚠️ **Moderate LIME stability**: Consider running multiple explanations")
                        else:
                            st.error("❌ **Low LIME stability**: Exercise caution with LIME explanations")
                
            except Exception as e:
                st.error(f"Error running stability audit: {str(e)}")
                st.write("Make sure SHAP and/or LIME explanations are enabled and working correctly.")
        
        else:
            st.info("Click 'Run Stability Audit' to test explanation robustness through input perturbations.")
            st.write("""
            **How it works:**
            1. Takes the selected instance and creates small variations by adding noise
            2. Computes explanations for each perturbed version  
            3. Measures how consistent the top features remain across runs
            4. Higher stability indicates more reliable explanations
            """)
        
except Exception as e:
    st.error(f"Error computing explanations: {str(e)}")
    st.info("This might be due to model compatibility issues. Please check the model and data formats.")
    
    # Show basic prediction info even if explanations fail
    st.subheader("Basic Prediction Info")
    st.write(f"Selected instance (row {idx}):")
    st.json(X_bg.iloc[idx].to_dict())
    st.write(f"Prediction probability: {proba[idx]:.4f}")
    st.write(f"Prediction class: {'> 50K' if proba[idx] > 0.5 else '<= 50K'}")



