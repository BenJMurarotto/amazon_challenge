# 📦 Multimodal Price Prediction with Pretrained Embeddings

This is my solution to a kaggle project to predict Amazon item prices based on images and catalogue content. I explore how multimodal machine learning can be used to improve predictive power. Instead of relying on a single input source, the model learns from product text descriptions, images, and structured features, reflecting how pricing decisions work in real-world e-commerce systems.

The focus is on practical multimodal modelling using pretrained embeddings rather than training large models from scratch.
All relevant CSV files and challenge details can be found at: https://www.kaggle.com/datasets/raghavdharwal/amazon-ml-challenge-2025/data

## 🧠 Approach

Pretrained models were used to extract rich semantic representations:

Text → SentenceTransformers (all-MiniLM-L12-v2)

Images → CLIP Vision Transformer (ViT-B/32)

Numeric features → Scaled quantity values and unit encodings

Several models were trained and evaluated under identical conditions:

📄 Text-only model

🖼️ Image-only model

🔗 Combined multimodal model (late fusion)

The multimodal model processes each modality independently and then fuses them in a shared regression head.

## 🏋️ Training & Evaluation

Targets were transformed using log(1 + price) to reduce skew and stabilise training

Models were trained with PyTorch using the Adam optimiser

Early stopping was applied based on validation loss

Evaluation was performed on a held-out test set using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MSE (Mean Squared Error)

All metrics are reported in USD

## 📊 Results

The combined multimodal model achieved the best overall performance, outperforming both unimodal baselines across all metrics.

Key observations:

Text and image models perform similarly, suggesting overlapping price signals

Multimodal fusion provides consistent (though modest) improvements

Errors increase substantially for high-priced and bulk items

