import streamlit as st
import pandas as pd
import io
import joblib
import os

from deployment.exporter import ExportManager

def show():
    st.header("Model Deployment & Export")

    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models first.")
    else:
        # Create tabs for deployment actions
        tab1, tab2, tab3, tab4 = st.tabs(["Deploy Models", "Deployed Models", "Export Models", "Download Models"])

        with tab1:
            st.subheader("Select Models to Deploy")

            # Show available trained models
            available_models = list(st.session_state.trained_models.keys())
            selected_for_deployment = st.multiselect(
                "Select models to deploy:",
                available_models,
                help="Choose which trained models to deploy",
                key="select_models_deploy"
            )

            if selected_for_deployment:
                st.subheader("Deployment Configuration")

                col1, col2 = st.columns(2)
                with col1:
                    deployment_name = st.text_input(
                        "Deployment Name:",
                        value=f"deployment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Name for this deployment"
                    )

                with col2:
                    export_formats = st.multiselect(
                        "Export Formats:",
                        ["joblib", "json", "png"],
                        default=["joblib"],
                        help="Formats to export the model in",
                        key="export_formats_deploy"
                    )

                # Show model details
                st.subheader("Model Details")
                for model_name in selected_for_deployment:
                    model_info = st.session_state.trained_models[model_name]
                    with st.expander(f" {model_name} Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Problem Type", model_info['problem_type'].capitalize())
                        with col2:
                            st.metric("Target Column", model_info['target_col'])
                        with col3:
                            if 'metrics' in model_info:
                                if model_info['problem_type'] == 'regression':
                                    score = model_info['metrics'].get('R2', 'N/A')
                                    st.metric("R² Score", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                elif model_info['problem_type'] == 'clustering':
                                    score = model_info['metrics'].get('Silhouette', 'N/A')
                                    st.metric("Silhouette", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                else:
                                    score = model_info['metrics'].get('Accuracy', 'N/A')
                                    st.metric("Accuracy", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                            else:
                                score = model_info.get('score', 'N/A')
                                st.metric("Score", f"{score:.4f}" if isinstance(score, (int, float)) else score)

                        st.write(f"**Features:** {', '.join(model_info['features'])}")

                if st.button("Deploy Selected Models", type="primary"):
                    try:
                        exporter = ExportManager()
                        deployed_count = 0

                        for model_name in selected_for_deployment:
                            model_info = st.session_state.trained_models[model_name]

                            # Prepare metadata
                            metadata = {
                                'deployment_name': deployment_name,
                                'model_name': model_name,
                                'problem_type': model_info['problem_type'],
                                'target_column': model_info['target_col'],
                                'features': model_info['features'],
                                'metrics': model_info.get('metrics', {}),
                                'deployed_at': pd.Timestamp.now().isoformat()
                            }

                            # Export model
                            export_result = exporter.export_model(
                                model=model_info['model'],
                                model_name=model_name,
                                preprocessing_pipeline=None,  # Could be added later
                                metadata=metadata,
                                formats=export_formats
                            )

                            # Store in deployed models dict
                            deployment_key = f"{deployment_name}_{model_name}"
                            st.session_state.deployed_models[deployment_key] = {
                                'model_info': model_info,
                                'export_paths': export_result,
                                'metadata': metadata,
                                'deployment_name': deployment_name
                            }

                            deployed_count += 1

                        st.success(f"✅ Successfully deployed {deployed_count} model(s)!")

                        

                    except Exception as e:
                        st.error(f"Deployment failed: {str(e)}")
                        st.code(str(e))

        with tab2:
            st.subheader("Currently Deployed Models")

            # Get all deployed models
            deployed_keys = list(st.session_state.deployed_models.keys())

            if not deployed_keys:
                st.info("No models currently deployed.")
            else:
                # Group by deployment name
                deployments = {}
                for key in deployed_keys:
                    try:
                        data = st.session_state.deployed_models.get(key)
                        dep_name = data.get('deployment_name', 'Unknown')
                        if dep_name not in deployments:
                            deployments[dep_name] = []
                        deployments[dep_name].append((key, data))
                    except:
                        continue

                for dep_name, models in deployments.items():
                    with st.expander(f"🚀 Deployment: {dep_name} ({len(models)} models)") as expander:
                        for key, data in models:
                            model_name = data['metadata']['model_name']
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                            with col1:
                                st.write(f"**{model_name}**")
                                st.caption(f"Target: {data['metadata']['target_column']}")

                            with col2:
                                problem_type = data['metadata']['problem_type']
                                st.metric("Type", problem_type.capitalize())

                            with col3:
                                if 'metrics' in data['model_info'] and data['model_info']['metrics']:
                                    if problem_type == 'regression':
                                        score = data['model_info']['metrics'].get('R2', 'N/A')
                                        st.metric("R² Score", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                    elif problem_type == 'clustering':
                                        score = data['model_info']['metrics'].get('Silhouette', 'N/A')
                                        st.metric("Silhouette", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                    else:
                                        score = data['model_info']['metrics'].get('Accuracy', 'N/A')
                                        st.metric("Accuracy", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                else:
                                    st.metric("Score", "N/A")

                            with col4:
                                if st.button(f"Delete", key=f"delete_{key}"):
                                    try:
                                        del st.session_state.deployed_models[key]
                                        st.success(f"Deleted {model_name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {str(e)}")

        with tab3:
            st.subheader("Export Deployed Models")

            # Get deployed models for export
            exportable_models = list(st.session_state.deployed_models.keys())

            if not exportable_models:
                st.info("No deployed models available for export.")
            else:
                selected_for_export = st.multiselect(
                    "Select deployed models to export:",
                    exportable_models,
                    help="Choose which deployed models to export",
                    key="select_models_export"
                )

                if selected_for_export:
                    export_formats = st.multiselect(
                        "Export Formats:",
                        ["joblib", "json", "png"],
                        default=["joblib"],
                        help="Formats to export the model in",
                        key="export_formats_export"
                    )

                    if st.button("Export Selected Models", type="primary"):
                        try:
                            exporter = ExportManager()
                            export_results = {}

                            for key in selected_for_export:
                                data = st.session_state.deployed_models.get(key)
                                model_info = data['model_info']
                                metadata = data['metadata']

                                # Export with additional formats if requested
                                result = exporter.export_model(
                                    model=model_info['model'],
                                    model_name=metadata['model_name'],
                                    preprocessing_pipeline=None,
                                    metadata=metadata
                                )

                                export_results[key] = result

                            st.success("Export completed!")

                            

                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                            st.code(str(e))

                    with tab4:
                        st.subheader("Download Deployed Models")

                        deployed_keys = list(st.session_state.deployed_models.keys())

                        if not deployed_keys:
                            st.info("No deployed models available for download.")
                        else:
                            for key in deployed_keys:
                                data = st.session_state.deployed_models.get(key)
                                if not data:
                                    continue

                                model_name = data['metadata'].get('model_name', key)
                                with st.expander(f"📥 {model_name} ({key})"):
                                    export_paths = data.get('export_paths', {})
                                    if not export_paths:
                                        st.write("No exported files available for this deployment.")
                                        continue

                                    for fmt, path in export_paths.items():
                                        try:
                                            if os.path.exists(path):
                                                with open(path, 'rb') as f:
                                                    file_bytes = f.read()

                                                btn_label = f"Download {model_name}.{fmt}"
                                                st.download_button(
                                                    label=btn_label,
                                                    data=file_bytes,
                                                    file_name=os.path.basename(path),
                                                    mime="application/octet-stream",
                                                    key=f"download_{key}_{fmt}"
                                                )
                                            else:
                                                st.write(f"{fmt.upper()}: {path} (file not found)")
                                        except Exception as e:
                                            st.error(f"Failed to prepare download for {model_name} ({fmt}): {e}")
                                            st.code(str(e))
