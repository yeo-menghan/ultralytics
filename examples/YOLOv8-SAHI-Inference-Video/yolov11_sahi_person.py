# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import logging
import platform
import time
from pathlib import Path

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


class SAHIInference:
    """
    Runs Ultralytics YOLO11 and SAHI for person detection on video with options to view, save, and track results.
    """

    def __init__(self):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("gpu_usage.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SAHIInference")

        # Check and log system information
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"PyTorch version: {torch.__version__}")

        # Check for GPU and log information
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")

        # Log GPU-specific info if available
        if self.device == "cuda:0":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif self.device == "mps":
            self.logger.info("Using Apple Silicon GPU via Metal Performance Shaders (MPS)")

    def _get_device(self):
        """
        Determine the best available device for inference.

        Returns:
            str: Device to use for inference ('cuda:0', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = "cuda:0"
            self.logger.info("CUDA is available! Using NVIDIA GPU.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
            self.logger.info("MPS (Metal Performance Shaders) is available! Using Apple GPU.")
        else:
            device = "cpu"
            self.logger.info("No GPU available. Using CPU.")

        return device

    def load_model(self, weights: str) -> None:
        """
        Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
        """
        yolo11_model_path = f"models/{weights}"
        download_yolo11n_model(yolo11_model_path)  # Download model if not present

        self.logger.info(f"Loading model from {yolo11_model_path} to {self.device}")

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=yolo11_model_path,
            device=self.device
        )

        self.logger.info(f"Model loaded successfully to {self.device}")

    def inference(
        self,
        weights: str = "yolo11n.pt",
        source: str = "test.mp4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """
        Run person detection on a video using YOLO11 and SAHI.

        Args:
            weights (str): Model weights path.
            source (str): Video file path.
            view_img (bool): Whether to display results in a window.
            save_img (bool): Whether to save results to a video file.
            exist_ok (bool): Whether to overwrite existing output files.
        """
        self.logger.info(f"Starting inference on {source} with {weights}")

        # Video setup
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"Error reading video file: {source}"
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video properties: {frame_width}x{frame_height} at {fps}fps, {total_frames} frames")

        # Output setup
        save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(save_dir / f"{Path(source).stem}.mp4")

        video_writer = None
        if save_img:
            self.logger.info(f"Output will be saved to {output_path}")
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # Using mp4v codec
                fps,
                (frame_width, frame_height),
            )

        # Load model
        self.load_model(weights)

        # Person class is typically class ID 0 in COCO dataset (which YOLO models use)
        person_class_id = 0

        frames_processed = 0
        start_time = time.time()

        # Memory monitoring for MPS (simplified)
        if self.device == "mps":
            # Check if the model is on MPS
            self.logger.info("Model running on Apple Silicon GPU via MPS")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frames_processed += 1
            if frames_processed % 10 == 0:
                self.logger.info(f"Processing frame {frames_processed}/{total_frames}")

            annotator = Annotator(frame)  # Initialize annotator for plotting detection results

            # Perform sliced prediction using SAHI
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.detection_model,
                slice_height=512,
                slice_width=512,
            )

            # Extract only person detections from results
            person_detections = []
            for det in results.object_prediction_list:
                if det.category.id == person_class_id or det.category.name.lower() == "person":
                    bbox = (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy)
                    # Convert score to float if needed
                    try:
                        conf_value = float(det.score.value)
                    except:
                        # If conversion fails, just use a string representation
                        conf_value = str(det.score)

                    person_detections.append((det.category.name, det.category.id, bbox, conf_value))

            # Annotate frame with detection results
            for name, id, bbox, conf in person_detections:
                # Include confidence score in the label if it's a float
                if isinstance(conf, float):
                    label = f"person {conf:.2f}"
                else:
                    label = f"person {conf}"
                annotator.box_label(bbox, label=label, color=colors(0, True))

            # Display results if requested
            if view_img:
                cv2.imshow(Path(source).stem, frame)

            # Save results if requested
            if save_img and video_writer:
                video_writer.write(frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Calculate and log performance metrics
        elapsed_time = time.time() - start_time
        self.logger.info(f"Inference completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Processed {frames_processed} frames")

        if frames_processed > 0 and elapsed_time > 0:
            self.logger.info(f"Average FPS: {frames_processed / elapsed_time:.2f}")

        # Clean up resources
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

        self.logger.info("Inference completed successfully")

    def parse_opt(self) -> argparse.Namespace:
        """
        Parse command line arguments for the inference process.

        Returns:
            (argparse.Namespace): Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolo11n.pt", help="initial weights path")
        parser.add_argument("--source", type=str, required=True, help="video file path")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
