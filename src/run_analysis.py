import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import load_metadata, create_splits, create_transformations, CassavaDataset
from model import create_model
from evaluate import predict, compute_metrics
from gradcam import GradCAM, create_overlay, prepare_gradcam_images

def select_gradcam_examples(df):
    """
    After reviewing the data provided by the model, we discovered four common confusion for models: CGM being mistaken
    for CMD, healthy cases being mistaken for CBB, healthy cases being mistaken for CGM, and CBSD being mistaken for
    CMD. This method grabs the highest confidence mistakes in each of these four categories, along with two of the
    highest confidence healthy and CMD cases for a baseline, and returns their information to be used for the
    GradCAM process.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the image names, correct classes, predicted classes, and confidence scores.

    Returns
    -------
    examples: pd.Series
        A DataFrame containing six examples of either common errors in the model or correct predictions.
    """
    examples = {}

    examples["CGM_CMD"] = (df[
                            (df["label"] == 2)
                            & (df["predicted_label"] == 3)]
                            .sort_values("confidence", ascending=False)
                            .iloc[0])

    examples["healthy_CBB"] = (df[
                                (df["label"] == 4)
                                & (df["predicted_label"] == 0)]
                                .sort_values("confidence", ascending=False)
                                .iloc[0])

    examples["healthy_CGM"] = (df[
                                (df["label"] == 4)
                                & (df["predicted_label"] == 2)]
                                .sort_values("confidence", ascending=False)
                                .iloc[0])

    examples["CBSD_CMD"] = (df[
                            (df["label"] == 1)
                            & (df["predicted_label"] == 3)]
                            .sort_values("confidence", ascending=False)
                            .iloc[0])

    examples["true_CMD"] = (df[
                            (df["label"] == 3)
                            & (df["predicted_label"] == 3)]
                            .sort_values("confidence", ascending=False)
                            .iloc[0])

    examples["true_healthy"] = (df[
                            (df["label"] == 4)
                            & (df["predicted_label"] == 4)]
                            .sort_values("confidence", ascending=False)
                            .iloc[0])

    return examples

def create_gradcam(gradcam:GradCAM, data:pd.DataFrame, simplified_labels:dict, image_dir:Path, device, model_transform):
    """
    This runs all six images found in the select_gradcam_examples method through the GradCAM process and generates the
    final comparison image with MatPlotLib.

    Parameters
    ----------
    gradcam: GradCAM
        A GradCAM object used to initialize the process, move the image through the model, and remove all hooks from
        the model

    data: pd.Series
        A DataFrame containing the image names, correct classes, predicted classes, and confidence scores to be used in
        the GradCAM process.

    simplified_labels: dict
        A dictionary containing the simplified disease class names (For example, "CGM" instead of "Cassava Green
        Mottle (CGM).")

    image_dir: Path
        A Path object pointing to the directory containing the images.

    device: torch.device
        A PyTorch device where the model is running.

    model_transform: torchvision.transforms.v2
        A v2 composition transformation used to transform the test images.
    """
    num = len(data)
    keys = list(data.keys())

    fig, axs = plt.subplots(num, 2, figsize=(5, 15))

    for i in range(num):
        current_key = keys[i]
        img_path = os.path.join(image_dir, data[current_key]["image_id"])

        display_image, model_img = prepare_gradcam_images(
            img_path,
            model_transform,
            device
        )

        heatmap, pred = gradcam.generate(model_img)

        display_img = np.array(display_image)
        overlay = create_overlay(
            heatmap,
            display_img
        )

        for j in range(axs.shape[1]):
            if j == 0:
                true_class = simplified_labels[int(data[current_key]["label"])]
                predicted_class = simplified_labels[int(data[current_key]["predicted_label"])]
                title_string = f"Original. True {true_class} | Predicted {predicted_class}"

                axs[i,j].imshow(display_img)
                axs[i,j].set_title(title_string)
            else:
                axs[i,j].imshow(overlay)
                axs[i,j].set_title("GradCAM")

            axs[i,j].axis('off')

    plt.show()

def main():
    data_dir = Path("data")
    checkpoint = Path("models/best_finetuned_resnet18.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    df, label_map, image_dir, simple_label = load_metadata(data_dir, return_simple=True)

    _, _, test_df = create_splits(df)
    _, _, test_transform = create_transformations()

    test_dataset = CassavaDataset(test_df, image_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers = 4,
                             persistent_workers=True, prefetch_factor=4)

    model = create_model(checkpoint=checkpoint)
    model = model.to(device)

    print("Running data through model...")
    y_true, y_pred, confidence = predict(model, test_loader, device)
    print("Prediction complete.\nCalculating metrics...")

    metrics = compute_metrics(y_true, y_pred, label_map, report_f1=True)

    result_df = test_df.reset_index(drop=True).copy()

    result_df["predicted_label"] = y_pred
    result_df["confidence"] = confidence
    result_df["correct"] = (result_df["predicted_label"] == result_df["label"])

    result_df.to_csv("results/test_predictions.csv", index = False)

    gradcam = GradCAM(model, model.layer4[-1].conv2)

    examples = select_gradcam_examples(result_df)

    create_gradcam(gradcam, examples, simple_label, image_dir, device, test_transform)

    gradcam.remove()

    print(f"Model test completed!")
    for key, val in metrics.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()

