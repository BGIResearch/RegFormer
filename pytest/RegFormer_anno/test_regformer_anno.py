# test_anno.py

import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path
import torch

# Add the project root to the Python path to ensure modules can be imported
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
from downstream_task.regformer_anno import AnnoTaskMamba

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_anno/anno_test.log"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH,
    filemode='w'
)
# Configure console output as well for real-time monitoring
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


class TestAnnoTaskPipeline(unittest.TestCase):

    # Using @classmethod decorator, these variables and methods belong to the class level.
    # They run only once before all tests start and clean up once after they finish.
    @classmethod
    def setUpClass(cls):
        """
        Execute the one-time, time-consuming setup work before all tests start.
        1. Load the real configuration.
        2. Set up a safe, temporary output directory.
        3. Initialize the AnnoTaskMamba instance.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting Annotation Task Test Suite " + "=" * 20)

        # --- 1. Load the real configuration ---
        real_config_filename = "anno_10k_SI.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / real_config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")
        config = toml.load(real_config_path)

        # --- 2. [Safety Measure] Set up a temporary output directory ---
        cls.test_dir = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_drug/temp_anno_test_output")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.test_dir)
        config['epochs'] = 1  # Force running only one epoch to speed up the test

        safe_config_path = cls.test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        # --- 3. Initialize the task instance ---
        cls.logger.info("STEP 0: Initializing AnnoTaskMamba instance...")
        cls.task = AnnoTaskMamba(config_file=safe_config_path)
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """Execute the one-time, thorough cleanup work after all tests have finished."""
        cls.logger.info("Cleaning up resources...")

        # Close and remove custom file log handlers ---
        if hasattr(cls.task, 'logger'):
            cls.logger.info("Closing custom file logger to release 'run.log'...")
            # Iterate over a copy of the handlers list to safely remove items
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)

        # Finally, try to remove the directory ---
        cls.logger.info("Cleaning up temporary directory...")
        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir)
                cls.logger.info("Temporary directory successfully removed.")
            except OSError as e:
                cls.logger.error(f"Failed to remove temporary directory: {e}. It might need to be manually deleted.")

        cls.logger.info("=" * 20 + " Annotation Task Test Suite Finished " + "=" * 20)

    def test_step_01_load_data_and_model(self):
        """Step 1: Verify that the data and model can be loaded successfully."""
        self.logger.info("--- Anno [Step 1/5]: Testing data and model loading ---")
        try:
            model, train_loader, valid_loader, test_loader, data_configs = self.task.load_data_and_model()
            # Save state for subsequent steps
            type(self).model = model
            type(self).train_loader = train_loader
            type(self).valid_loader = valid_loader
            type(self).test_loader = test_loader
            type(self).data_configs = data_configs

            self.assertIsNotNone(self.model, "Model object should not be None.")
            self.assertIsNotNone(self.train_loader, "Train loader should not be None.")
            self.logger.info("--- Anno [Step 1/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Anno [Step 1/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 1 failed: {e}")

    def test_step_02_criterion_and_optimizer_setup(self):
        """Step 2: Verify that the loss function and optimizer can be created successfully."""
        self.logger.info("--- Anno [Step 2/5]: Testing criterion and optimizer setup ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")
            self.task._loadCriterionAndOpt(self.model)
            self.assertIsNotNone(self.task.criterion_cls, "Criterion should not be None.")
            self.assertIsNotNone(self.task.optimizer, "Optimizer should not be None.")
            self.logger.info("--- Anno [Step 2/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Anno [Step 2/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 2 failed: {e}")

    def test_step_03_train_and_evaluate_one_epoch(self):
        """Step 3: Verify that the model can complete one epoch of training and evaluation."""
        self.logger.info("--- Anno [Step 3/5]: Testing training and evaluation for one epoch ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")
            self.task._train(self.model, self.train_loader, epoch=1)

            # This logic is from the main runAnnotation loop
            best_f1 = 0.0
            _, _, _, _, _, f1 = self.task._evaluate(self.model, self.valid_loader, epoch=1)
            if f1 > best_f1:
                type(self).best_model = self.model

            self.assertIsNotNone(self.best_model, "Best model should be set after evaluation.")
            self.logger.info("--- Anno [Step 3/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Anno [Step 3/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 3 failed: {e}")

    def test_step_04_final_test_and_organize_results(self):
        """Step 4: Verify the final test and result organization stages."""
        self.logger.info("--- Anno [Step 4/5]: Testing with the final model and organizing results ---")
        try:
            self.assertTrue(hasattr(self, 'best_model'), "Best model from step 3 is missing.")
            preds, labels, results_dict = self.task._test(self.best_model, self.test_loader,
                                                          self.data_configs["test_labels"])
            self.task._organizeResults(preds, labels, results_dict, self.data_configs)

            results_file = self.task.save_dir / "results.pkl"
            self.assertTrue(results_file.exists(), "results.pkl file was not created.")
            self.logger.info("--- Anno [Step 4/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Anno [Step 4/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 4 failed: {e}")

    def test_step_05_save_model(self):
        """Step 5: Verify that the best model can be saved successfully."""
        self.logger.info("--- Anno [Step 5/5]: Testing model saving ---")
        try:
            self.assertTrue(hasattr(self, 'best_model'), "Best model from step 3 is missing.")
            torch.save(self.best_model.state_dict(), self.task.save_dir / "best_model.pt")
            self.assertTrue((self.task.save_dir / "best_model.pt").exists())
            self.logger.info("--- Anno [Step 5/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Anno [Step 5/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 5 failed: {e}")


if __name__ == '__main__':
    # Redirect all unittest runner output (including tracebacks) to the log file.
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestAnnoTaskPipeline))
        # Direct the runner's output stream to our log file.
        runner = unittest.TextTestRunner(verbosity=2, stream=log_file)
        print(f"Running tests for Annotation task. See detailed logs in {LOG_FILE_PATH}")
        runner.run(suite)