import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path
import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the Python path to ensure modules can be imported
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
from downstream_task.regformer_drug import DrugTaskMamba
from regformer.repo.deepcdr.drug_data_process import DrugDataProcess
from regformer.repo.deepcdr.drug import PyTorchMultiSourceGCNModel

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_drug/drug_test.log"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE_PATH,
    filemode='w'
)
# Configure console output as well, for real-time monitoring
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Add handler to the root logger
logging.getLogger().addHandler(console_handler)


class TestDrugTaskDetailed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the testing environment once for all tests in this class.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting Detailed Drug Response Task Test Suite " + "=" * 20)

        real_config_filename = "drug_10k.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / real_config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")
        config = toml.load(real_config_path)

        cls.test_dir = Path("./temp_test_output_drug_detailed")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.test_dir)
        config['epochs'] = 1

        safe_config_path = cls.test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        cls.logger.info("STEP 0: Initializing DrugTaskMamba instance...")
        cls.task = DrugTaskMamba(config_file=safe_config_path)
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources after all tests have run.
        """
        cls.logger.info("Cleaning up resources for Drug Response Task...")
        if hasattr(cls, 'task') and hasattr(cls.task, 'logger'):
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        cls.logger.info("=" * 20 + " Detailed Drug Response Task Test Suite Finished " + "=" * 20)

    def test_step_01_metadata_generation(self):
        """Step 1: Test loading and preprocessing of all raw data files."""
        self.logger.info("--- Drug [Step 1/6]: Testing Metadata Generation ---")
        try:
            data_obj = DrugDataProcess(self.task.args)
            mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = data_obj.MetadataGenerate(
                self.task.args.drug_info_file, self.task.args.cell_line_info_file, self.task.args.genomic_mutation_file,
                self.task.args.drug_feature_file, self.task.args.gene_expression_file, self.task.args.methylation_file)

            type(self).data_obj = data_obj
            type(self).mutation_feature = mutation_feature
            type(self).drug_feature = drug_feature
            type(self).gexpr_feature = gexpr_feature
            type(self).methylation_feature = methylation_feature
            type(self).data_idx = data_idx

            self.assertIsInstance(gexpr_feature, pd.DataFrame)
            self.assertFalse(gexpr_feature.empty)
            self.logger.info("--- Drug [Step 1/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 1/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 1 failed: {e}")

    def test_step_02_pretrain_inference(self):
        """Step 2: Test generation of gene expression embeddings."""
        self.logger.info("--- Drug [Step 2/6]: Testing Pre-trained Model Inference (Embedding) ---")
        try:
            self.assertTrue(hasattr(self, 'gexpr_feature'), "GExpr feature from step 1 is missing.")
            gexpr_emb = self.task.pretrain_inference(self.gexpr_feature)
            type(self).gexpr_emb = gexpr_emb
            self.assertIsInstance(gexpr_emb, pd.DataFrame)
            self.assertEqual(gexpr_emb.shape[0], self.gexpr_feature.shape[0])
            self.logger.info("--- Drug [Step 2/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 2/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 2 failed: {e}")

    def test_step_03_data_split(self):
        """Step 3: Test splitting of data indices into train/test sets."""
        self.logger.info("--- Drug [Step 3/6]: Testing Data Splitting ---")
        try:
            self.assertTrue(hasattr(self, 'data_obj'), "Data object from step 1 is missing.")
            data_train_idx, data_test_idx = self.data_obj.DataSplit(self.data_idx)
            type(self).data_train_idx = data_train_idx
            type(self).data_test_idx = data_test_idx
            self.assertGreater(len(data_train_idx), 0)
            self.assertGreater(len(data_test_idx), 0)
            self.logger.info("--- Drug [Step 3/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 3/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 3 failed: {e}")

    def test_step_04_feature_extraction(self):
        """Step 4: Test extraction of features into tensors for train and test sets."""
        self.logger.info("--- Drug [Step 4/6]: Testing Feature Extraction ---")
        try:
            self.assertTrue(hasattr(self, 'data_train_idx'), "Train indices from step 3 are missing.")

            X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, _ = self.data_obj.FeatureExtract(
                self.data_train_idx, self.drug_feature, self.mutation_feature, self.gexpr_emb, self.methylation_feature)

            X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, _ = self.data_obj.FeatureExtract(
                self.data_test_idx, self.drug_feature, self.mutation_feature, self.gexpr_emb, self.methylation_feature)

            # Add the missing torch.stack operations to match the main script
            X_drug_feat_data_test = torch.stack([item[0] for item in X_drug_data_test])
            X_drug_adj_data_test = torch.stack([item[1] for item in X_drug_data_test])

            type(self).X_drug_data_train = X_drug_data_train
            type(self).X_mutation_data_train = X_mutation_data_train
            type(self).X_gexpr_data_train = X_gexpr_data_train
            type(self).X_methylation_data_train = X_methylation_data_train
            type(self).Y_train = Y_train
            type(self).validation_data = [
                [X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test,
                 X_methylation_data_test], Y_test]

            self.assertIsInstance(Y_train, torch.Tensor)
            self.assertIsInstance(Y_test, torch.Tensor)
            self.logger.info("--- Drug [Step 4/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 4/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 4 failed: {e}")

    def test_step_05_model_and_optimizer_initialization(self):
        """Step 5: Test the initialization of the GCN model and Adam optimizer."""
        self.logger.info("--- Drug [Step 5/6]: Testing Model & Optimizer Initialization ---")
        try:
            self.assertTrue(hasattr(self, 'X_drug_data_train'), "Train data from step 4 is missing.")

            model = PyTorchMultiSourceGCNModel(
                drug_input_dim=self.X_drug_data_train[0][0].shape[-1], drug_hidden_dim=256, drug_concate_before_dim=100,
                mutation_input_dim=self.X_mutation_data_train.shape[-2], mutation_hidden_dim=256,
                mutation_concate_before_dim=100,
                gexpr_input_dim=self.X_gexpr_data_train.shape[-1], gexpr_hidden_dim=256, gexpr_concate_before_dim=100,
                methy_input_dim=self.X_methylation_data_train.shape[-1], methy_hidden_dim=256,
                methy_concate_before_dim=100,
                output_dim=300, units_list=self.task.args.unit_list, use_mut=self.task.args.use_mut,
                use_gexp=self.task.args.use_gexp,
                use_methy=self.task.args.use_methy, regr=True, use_relu=self.task.args.use_relu,
                use_bn=self.task.args.use_bn,
                use_GMP=self.task.args.use_GMP
            ).to(self.task.device)
            optimizer = Adam(model.parameters(), lr=self.task.args.lr, eps=1e-07)

            type(self).model = model
            type(self).optimizer = optimizer

            self.assertIsNotNone(self.model)
            self.assertIsNotNone(self.optimizer)
            self.logger.info("--- Drug [Step 5/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 5/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 5 failed: {e}")

    def test_step_06_train_and_test(self):
        """Step 6: Test the model training loop and final evaluation."""
        self.logger.info("--- Drug [Step 6/6]: Testing Model Training and Final Test ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 5 is missing.")

            # This stacking is for the training dataloader
            X_drug_feat_data_train = torch.stack([item[0] for item in self.X_drug_data_train])
            X_drug_adj_data_train = torch.stack([item[1] for item in self.X_drug_data_train])

            train_data = TensorDataset(X_drug_feat_data_train, X_drug_adj_data_train, self.X_mutation_data_train,
                                       self.X_gexpr_data_train, self.X_methylation_data_train, self.Y_train)
            dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

            self.task.train(self.model, dataloader, self.validation_data, self.optimizer)

            results_file = self.task.save_dir / "random_test.csv"
            self.assertTrue(results_file.exists())
            self.logger.info("--- Drug [Step 6/6]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Drug [Step 6/6]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 6 failed: {e}")


if __name__ == '__main__':
    # Redirect unittest runner's output to the log file
    # This ensures that the final summary, including tracebacks, is written to the log.
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestDrugTaskDetailed))
        # Direct the runner's output stream to our log file
        runner = unittest.TextTestRunner(verbosity=2, stream=log_file)
        print(f"Running tests for Drug task. See detailed logs in {LOG_FILE_PATH}")
        runner.run(suite)