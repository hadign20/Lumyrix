# projects/rectal_wholemount/main.py

import os
import sys
import logging
import argparse
from pathlib import Path

def setup_logging(log_dir: str = "logs"):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "rectal_wholemount.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Rectal Wholemount PCR Prediction")
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Run the training pipeline"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Run the validation pipeline"
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true",
        help="Only run feature analysis"
    )
    
    parser.add_argument(
        "--select", 
        action="store_true",
        help="Only run feature selection"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=r"D:\projects\Lumyrix\projects\rectal_wholemount\config",
        help="Path to the configuration directory"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the rectal_wholemount project.
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load project configuration
    config_path = args.config
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration directory not found: {config_path}")
        return
    
    logger.info("Starting Rectal Wholemount PCR Prediction project")
    
    # Import radiomics pipeline
    from pipelines.radiomics_pipeline import RadiomicsPipeline
    
    # Initialize pipeline
    pipeline = RadiomicsPipeline(config_path)
    
    # Run requested pipeline components
    if args.analyze:
        logger.info("Running feature analysis only")
        # Load data
        train_data, _ = pipeline._load_data()
        
        # Run feature analysis
        for dataset_name, df in train_data.items():
            logger.info(f"Analyzing features in dataset: {dataset_name}")
            
            from core.feature_analysis.feature_analysis import FeatureAnalysis
            feature_analysis = FeatureAnalysis(config_path)
            
            # Create output directory for this dataset
            output_dir = os.path.join(pipeline.result_path, dataset_name, "feature_analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run analysis
            feature_analysis.analyze_features(df)
    
    elif args.select:
        logger.info("Running feature selection only")
        # Load data
        train_data, _ = pipeline._load_data()
        
        # Run feature selection
        for dataset_name, df in train_data.items():
            logger.info(f"Selecting features in dataset: {dataset_name}")
            
            from core.feature_selection.feature_selector import FeatureSelector
            feature_selector = FeatureSelector(config_path)
            
            # Create output directory for this dataset
            output_dir = os.path.join(pipeline.result_path, dataset_name, "feature_selection")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run feature selection
            feature_selector.run_feature_selection(df)
    
    elif args.validate:
        logger.info("Running validation pipeline")
        pipeline._validate_models(None, None)
    
    elif args.train:
        logger.info("Running training pipeline")
        # Run the complete training pipeline
        pipeline.run()
    
    else:
        logger.info("Running complete pipeline (training and validation)")
        # Run the complete pipeline
        pipeline.run()
    
    logger.info("Rectal Wholemount PCR Prediction completed successfully")

if __name__ == "__main__":
    main()