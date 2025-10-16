import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path
import torch

# Add the project root to the Python path
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
from downstream_task.regformer_pert import PertTaskMamba
from regformer.repo.gears import PertData, GEARS

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_pert/pert_test.log"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH,
    filemode='w'
)
# Configure console output as well
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


class TestPertTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the testing environment once for all tests in this class.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting Perturbation Task Test Suite " + "=" * 20)

        real_config_filename = "pert_adamson_10k.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / real_config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")
        config = toml.load(real_config_path)

        cls.test_dir = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_pert/temp_test_output_pert")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.test_dir)
        config['epochs'] = 1

        safe_config_path = cls.test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        cls.logger.info("STEP 0: Initializing PertTaskMamba instance...")
        cls.task = PertTaskMamba(config_file=safe_config_path)
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources once after all tests in this class have run.
        """
        cls.logger.info("Cleaning up resources for Perturbation Task...")
        if hasattr(cls.task, 'logger'):
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)
        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir)
                cls.logger.info("Temporary directory successfully removed.")
            except OSError as e:
                cls.logger.error(f"Failed to remove temporary directory: {e}.")

        cls.logger.info("=" * 20 + " Perturbation Task Test Suite Finished " + "=" * 20)

    def test_step_01_make_pert_data(self):
        """Step 1: Verify that perturbation data can be loaded and processed."""
        self.logger.info("--- Pert [Step 1/4]: Testing perturbation data preparation ---")
        try:
            pert_data = self.task.make_pert_data()
            type(self).pert_data = pert_data
            self.assertIsInstance(self.pert_data, PertData, "Output should be a PertData object.")
            self.assertIn('train_loader', self.pert_data.dataloader, "Dataloader should contain a train_loader.")
            self.logger.info("--- Pert [Step 1/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pert [Step 1/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 1 failed: {e}")

    def test_step_02_get_gene_embedding_matrix(self):
        """Step 2: Verify that the universal gene embedding matrix can be created."""
        self.logger.info("--- Pert [Step 2/4]: Testing universal gene embedding creation ---")
        try:
            self.assertTrue(hasattr(self, 'pert_data'), "PertData from previous step is missing.")
            gene_emb_weight = self.task.universal_gene_embedding(self.pert_data)
            type(self).gene_emb_weight = gene_emb_weight
            self.assertIsInstance(self.gene_emb_weight, torch.Tensor, "Embedding weight should be a torch.Tensor.")
            self.assertEqual(self.gene_emb_weight.shape[0], len(self.pert_data.gene_names))
            self.logger.info("--- Pert [Step 2/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pert [Step 2/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 2 failed: {e}")

    def test_step_03_initialize_and_train_gears_model(self):
        """Step 3: Verify that the GEARS model can be initialized, trained, and saved."""
        self.logger.info("--- Pert [Step 3/4]: Testing GEARS model initialization and training ---")
        try:
            self.assertTrue(hasattr(self, 'pert_data'), "PertData from step 1 is missing.")
            self.assertTrue(hasattr(self, 'gene_emb_weight'), "Gene embedding matrix from step 2 is missing.")

            model = GEARS(
                self.pert_data,
                device=self.task.device,
                model_output=str(self.task.save_dir)
            )

            model.model_initialize(
                hidden_size=self.task.args.hidden_size,
                use_pretrained=self.task.args.use_pretrained,
                pretrain_freeze=self.task.args.pretrain_freeze,
                gene_emb_weight=self.gene_emb_weight,
                pretrained_emb_size=self.task.args.layer_size,
                gene2ids=self.task.gene2ids,
                layer_size=self.task.args.layer_size
            )
            model.best_model.to(self.task.device)

            if self.task.args.finetune:
                model.train(epochs=self.task.args.epochs, lr=self.task.args.lr)
                self.logger.info("Saving the best model...")
                model.save_model(str(self.task.save_dir))

            type(self).gears_model = model

            self.assertIsInstance(self.gears_model, GEARS, "A GEARS model object should be created.")
            if self.task.args.finetune:
                self.assertTrue((self.task.save_dir / "model.pt").exists(),
                                "Best model was not saved during training.")
            self.logger.info("--- Pert [Step 3/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pert [Step 3/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 3 failed: {e}")

    def test_step_04_evaluate_and_check_output(self):
        """Step 4: Verify that the trained model can be evaluated and results saved."""
        self.logger.info("--- Pert [Step 4/4]: Testing model evaluation ---")
        try:
            self.assertTrue(hasattr(self, 'pert_data'), "PertData from step 1 is missing.")
            self.assertTrue(hasattr(self, 'gears_model'), "GEARS model from step 3 is missing.")

            output_file_path = self.task.save_dir / "result.pkl"
            self.task.evaluate_pert(self.pert_data, self.gears_model, output_file=output_file_path)

            self.assertTrue(output_file_path.exists(), "Final result.pkl file was not created.")
            self.logger.info("--- Pert [Step 4/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Pert [Step 4/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 4 failed: {e}")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPertTask))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)