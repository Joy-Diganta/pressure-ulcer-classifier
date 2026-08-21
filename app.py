import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go

CurrentModel = "best_model_fold0.pth" # Don't change or rename it.

st.set_page_config(
    page_title="Pressure Ulcer Stage Classifier",
    page_icon="🩺",
    layout="centered"
)

st.markdown(
    """
    <style>
        /* =====================================================
           EXISTING DESKTOP STYLING — KEEP
           ===================================================== */

        /* Center text throughout the main application */
        .block-container {
            text-align: center;
        }

        /* Center success message */
        [data-testid="stAlert"] {
            text-align: center;
        }

        /* Center Plotly chart */
        [data-testid="stPlotlyChart"] {
            display: flex;
            justify-content: center;
        }


        /* =====================================================
           MOBILE RESPONSIVE STYLING
           ===================================================== */

        @media (max-width: 768px) {

            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
                width: 100% !important;
                max-width: 100% !important;
                text-align: center;
            }

            /* Main headings */
            h1 {
                font-size: 2rem !important;
                line-height: 1.15 !important;
        
                /* Critical for mobile */
                white-space: normal !important;
                overflow-wrap: break-word !important;
                word-break: normal !important;
        
                width: 100% !important;
                max-width: 100% !important;
        
                text-align: center !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            h2 {
                font-size: 1.5rem !important;
                line-height: 1.15 !important;
            }

            h3 {
                font-size: 1.2rem !important;
            }

            /* Body text */
            p,
            .stMarkdown {
                font-size: 1rem !important;
                line-height: 1.45 !important;
            }

            /* Images must never exceed phone width */
            img {
                max-width: 100% !important;
                height: auto !important;
            }

            /* Uploaded image */
            [data-testid="stImage"] {
                width: 100% !important;
                display: flex !important;
                justify-content: center !important;
            }

            /* File uploader */
            [data-testid="stFileUploader"] {
                width: 100% !important;
                max-width: 100% !important;
            }

            /* Success / alert messages */
            [data-testid="stAlert"] {
                width: 100% !important;
                max-width: 100% !important;
                text-align: center !important;
            }

            /* Metric */
            [data-testid="stMetric"] {
                width: 100% !important;
                max-width: 100% !important;
            }

            /* Plotly chart */
            [data-testid="stPlotlyChart"] {
                width: 100% !important;
                max-width: 100% !important;
                display: block !important;
            }

            /* Prevent horizontal overflow */
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewBlockContainer"],
            .main {
                max-width: 100% !important;
                width: 100% !important;
                overflow-x: hidden !important;
            }

            /* Footer / long text */
            footer,
            .footer {
                max-width: 100% !important;
                width: 100% !important;
                overflow-wrap: anywhere !important;
                word-wrap: break-word !important;
                text-align: center !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Pressure Ulcer Stage Classifier 📊", text_alignment = "center")
st.subheader("A Deep Learning Prototype 🔬", divider = "red", text_alignment = "center")
st.write("Upload a clinical photograph to check the classification stage.")

CLASS_NAMES = ['Invalid', 'SDTI', 'Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV', 'Unstageable']

# Load the Model Architecture (Cached so it only loads once)
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "The 'timm' package is required for ConvNeXt-V2-B. "
            "Install it with: pip install timm"
        ) from e

    checkpoint = torch.load(CurrentModel, map_location=device, weights_only=False)

    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model check if {CurrentModel} is in the folder. Details: {e}")

# Image Preprocessing Pipeline 
img_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Choose an image file...", 
                                 type=["jpg", "jpeg", "png"], 
                                 max_upload_size=15, 
                                 label_visibility="collapsed"
                                 )

col_1, col_2, col_3 = st.columns([1, 3, 1]) #For aligning the image to center

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    with col_2:
        st.image(image, caption="Uploaded Image", width="stretch", use_container_width=True)
    
    with st.spinner("Analyzing image... Please wait..."):
        input_tensor = img_transform(image).unsqueeze(0).to(torch.device("cpu"))

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        # Display Results
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_percentage = confidence.item() * 100

        st.success("Analysis Complete!")
        st.write("**Predicted Assessment:**")
        st.metric(
            label=" ",
            value=predicted_class,
            border=True,
            label_visibility="collapsed"
        )

        st.subheader(
            f"Confidence Level: {confidence_percentage:.2f}%",
            divider="gray", 
            text_alignment = "center"
        )

        bar_colors = [
            "#2A7F7F",
            "#4F81A8",
            "#6E9B78",
            "#C46F5A",
            "#80648F",
            "#C49A4A",
            "#B77B82"
        ]

        fig = go.Figure(
            go.Bar(
                x=CLASS_NAMES,
                y=[float(probabilities[i]) for i in range(len(CLASS_NAMES))],
                marker_color=bar_colors,
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>"
            )
        )

        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=20, b=110),
            xaxis=dict(
                title=None,
                tickangle=-45,
                automargin=True
            ),
            yaxis=dict(
                title="Probability",
                range=[0, 1],
                tickformat=".0%",
                automargin=True
            ),
            plot_bgcolor="#faf4d7",
            paper_bgcolor="#faf4d7",
            font=dict(
                family="serif",
                color="#2F2A24"
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid #D8CDAA;
        color: #6F6658;
        font-size: 15px;
        line-height: 1.6;
    ">
        © 2026 BME 310 Machine Learning Project<br>
        Department of Biomedical Engineering, BUET<br>
        <b>Disclaimer:</b> Academic research prototype for educational purposes only.<br>
        <span style="
            display: inline-block;
            color: #A24E3B;
            font-weight: 700;
        ">
            Not intended for clinical diagnosis or medical decision-making.
        </span><br>
        
    </div>
    """,
    unsafe_allow_html=True
)

@st.dialog("Team info", width="medium")
def project_info():
    st.image(
        "project_info.jpg",
        use_container_width=True
    )
    st.markdown("""
        <div style="
            text-align: center;
            line-height: 2;
        "><span style="font-size: 20px;">
                📧
                <a href="mailto:joydiganta.bd@gmail.com"
                   style="
                       color: #2A6F72;
                       font-weight: 600;
                       text-decoration: underline;
                       text-underline-offset: 3px;
                   ">
                    Contact project team
                </a>
            </span>
            </div>
                """,
                unsafe_allow_html=True
            )

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("ℹ️ Team info", type="secondary"):
     project_info()
