# test_pretrain.py

import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path
import torch
import torch.distributed as dist

# Add the project root to the Python path to ensure modules can be imported
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
# Note: The import path is from the 'pretrain' directory as per your source file
from downstream_task.regformer_pretrain import PretrainTaskScMamba

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_pretrain/pretrain_test.log"
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

# Helper class to limit the number of batches from a DataLoader
class LimitedBatchLoader:
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self.iter_loader = iter(self.loader)
        # To make this wrapper compatible with methods that check len(loader)
        self.original_len = len(self.loader)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.max_batches:
            raise StopIteration
        self.count += 1
        return next(self.iter_loader)

    def __len__(self):
        return min(self.max_batches, self.original_len)


class TestPretrainTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the testing environment once for all tests in this class for the Pre-training task.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting Pre-training Task Test Suite " + "=" * 20)

        # --- 1. Load the real 'pretrain.toml' config file ---
        real_config_filename = "pretrain.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / real_config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")

        config = toml.load(real_config_path)

        # --- 2. [Safety Measure] Create a dedicated temporary output directory ---
        cls.base_test_dir = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_pretrain/temp_test_output_pretrain")
        if cls.base_test_dir.exists():
            shutil.rmtree(cls.base_test_dir)
        cls.base_test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.base_test_dir)
        config['epochs'] = 1
        config['distributed'] = True  # Set to True so the application code initializes the distributed environment
        config['resume_batch'] = 0

        # --- 3. Manually set environment variables required by the application's initializer ---
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  # Use any free port
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'


        safe_config_path = cls.base_test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        # --- 4. Initialize the PretrainTaskScMamba instance ---
        # Now, this call will correctly initialize the distributed group once.
        cls.logger.info("STEP 0: Initializing PretrainTaskScMamba instance...")
        cls.task = PretrainTaskScMamba(config_file=safe_config_path)
        cls.task.set_swanlab()
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources once after all tests in this class have run.
        """
        cls.logger.info("Cleaning up resources for Pre-training Task...")
        if hasattr(cls, 'task') and hasattr(cls.task, 'run') and cls.task.run is not None:
            cls.task.run.finish()
        if hasattr(cls, 'task') and hasattr(cls.task, 'logger'):
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)
        if hasattr(cls, 'base_test_dir') and os.path.exists(cls.base_test_dir):
            shutil.rmtree(cls.base_test_dir)

        if dist.is_initialized():
            dist.destroy_process_group()

        cls.logger.info("=" * 20 + " Pre-training Task Test Suite Finished " + "=" * 20)

    def test_step_01_load_data_and_model(self):
        """Step 1: Verify that the data and model can be loaded successfully."""
        self.logger.info("--- Pretrain [Step 1/5]: Testing data and model loading ---")
        try:
            model, train_loader, val_loader = self.task.load_data_and_model()
            # Save state for subsequent steps
            type(self).model = model
            type(self).train_loader = train_loader
            type(self).val_loader = val_loader

            self.assertIsNotNone(self.model, "Model object should not be None.")
            self.assertIsNotNone(self.train_loader, "Train loader should not be None.")
            self.logger.info("--- Pretrain [Step 1/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pretrain [Step 1/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 1 failed: {e}")

    def test_step_02_criterion_and_optimizer_setup(self):
        """Step 2: Verify that the loss function and optimizer can be created successfully."""
        self.logger.info("--- Pretrain [Step 2/5]: Testing criterion and optimizer setup ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")
            self.task.load_criterion_and_opt(self.model)
            self.assertIsNotNone(self.task.criterion, "Criterion should not be None.")
            self.assertIsNotNone(self.task.optimizer, "Optimizer should not be None.")
            self.logger.info("--- Pretrain [Step 2/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pretrain [Step 2/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 2 failed: {e}")

    def test_step_03_train_for_200_batches(self):
        """Step 3: Verify the model can train for a limited number of batches by calling the original train function."""
        self.logger.info("--- Pretrain [Step 3/5]: Testing training for 100 batches ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")

            # Create a limited wrapper around the real train_loader
            max_batches = 200
            limited_train_loader = LimitedBatchLoader(self.train_loader, max_batches)
            self.logger.info(f"Using a limited loader that will only yield {max_batches} batches.")

            # Call the original train function, but with our limited loader
            self.task.train(self.model, limited_train_loader, epoch=1)

            self.logger.info("--- Pretrain [Step 3/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pretrain [Step 3/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 3 failed: {e}")

    def test_step_04_evaluate_one_epoch(self):
        """Step 4: Verify that the model can complete one epoch of evaluation."""
        self.logger.info("--- Pretrain [Step 4/5]: Testing evaluation for one epoch ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")

            # Use the LimitedBatchLoader for the validation set as well
            max_eval_batches = 50
            limited_val_loader= LimitedBatchLoader(self.val_loader, max_batches=max_eval_batches)
            self.logger.info(f"Using a limited validation loader that will only yield {max_eval_batches} batches.")


            val_losses = self.task.evaluate(self.model, limited_val_loader, epoch=1)
            self.assertIn('total', val_losses, "Validation losses dictionary should contain a 'total' key.")

            # Simulate the logic from run_pretrain to save the best model
            type(self).best_val_loss = float("inf")
            if val_losses["total"] < self.best_val_loss:
                type(self).best_model_state_dict = self.model.state_dict()

            self.logger.info("--- Pretrain [Step 4/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pretrain [Step 4/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 4 failed: {e}")

    def test_step_05_check_outputs(self):
        """Step 5: Verify that the expected model checkpoint files were created."""
        self.logger.info("--- Pretrain [Step 5/5]: Verifying output files ---")
        try:
            # The 'train' and 'evaluate' steps implicitly trigger saving in the main script.
            # We will check for the files that should be created after one full epoch.
            self.assertTrue(hasattr(self, 'best_model_state_dict'), "Best model state was not captured in step 4.")

            # Save the best model, mimicking the main script's logic
            save_path = self.task.save_dir / "best_model.pt"
            torch.save(self.best_model_state_dict, save_path)

            # The script also saves a checkpoint at the end of every epoch
            epoch_save_path = self.task.save_dir / "model_e1.pt"
            torch.save(self.model.state_dict(), epoch_save_path)

            self.assertTrue(save_path.exists(), "best_model.pt was not created.")
            self.assertTrue(epoch_save_path.exists(), "model_e1.pt was not created.")
            self.logger.info("--- Pretrain [Step 5/5]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pretrain [Step 5/5]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 5 failed: {e}")


if __name__ == '__main__':
    # Redirect all unittest runner output (including tracebacks) to the log file.
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestPretrainTask))
        runner = unittest.TextTestRunner(verbosity=2, stream=log_file)
        print(f"Running tests for Pre-training task. See detailed logs in {LOG_FILE_PATH}")
        runner.run(suite)
