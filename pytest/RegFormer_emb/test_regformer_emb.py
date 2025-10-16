import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm

# Add the project root to the Python path to ensure modules can be imported
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
from downstream_task.regformer_emb import EmbTaskMamba
from regformer.utils.utils import refine_embedding

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_emb/emb_test.log"
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
logging.getLogger().addHandler(console_handler)


class TestEmbTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the testing environment once for all tests in this class for the Embedding task.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting Embedding Task Test Suite " + "=" * 20)

        # --- 1. Load the real configuration for the Embedding task ---
        config_filename = "cell_emb_human_lung.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")

        config = toml.load(real_config_path)

        # --- 2. [Safety Measure] Create a dedicated temporary output directory ---
        cls.test_dir = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_emb/temp_test_output_emb")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.test_dir)
        config['epochs'] = 1  # The emb task does not involve training, but keeping this is good practice.

        safe_config_path = cls.test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        # --- 3. Initialize the EmbTaskMamba instance ---
        cls.logger.info("STEP 0: Initializing EmbTaskMamba instance...")
        cls.task = EmbTaskMamba(config_file=safe_config_path)
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources once after all tests in this class have run.
        """
        cls.logger.info("Cleaning up resources for Embedding Task...")
        if hasattr(cls, 'task') and hasattr(cls.task, 'logger'):
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        cls.logger.info("=" * 20 + " Embedding Task Test Suite Finished " + "=" * 20)

    def test_step_01_load_data_and_model(self):
        """Step 1: Verify that the data and model can be loaded successfully."""
        self.logger.info("--- Emb [Step 1/4]: Testing data and model loading ---")
        try:
            model, loader = self.task.load_data_and_model()
            # Save state for subsequent steps
            type(self).model = model
            type(self).loader = loader

            self.assertIsNotNone(self.model, "Model object should not be None.")
            self.assertIsNotNone(self.loader, "DataLoader should not be None.")
            self.logger.info("--- Emb [Step 1/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Emb [Step 1/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 1 failed: {e}")

    def test_step_02_run_inference_loop(self):
        """Step 2: Verify that the model inference loop can execute successfully."""
        self.logger.info("--- Emb [Step 2/4]: Testing inference loop to generate embeddings ---")
        try:
            self.assertTrue(hasattr(self, 'model'), "Model from step 1 is missing.")
            self.assertTrue(hasattr(self, 'loader'), "Loader from step 1 is missing.")

            # This logic replicates the core inference loop from run_embedding()
            self.model.eval()
            embeddings = np.zeros((len(self.loader.dataset), self.task.args.layer_size), dtype=np.float32)
            all_batch_labels = []
            all_celltype_labels = []
            count = 0
            with torch.no_grad():
                for batch in tqdm(self.loader, desc="Testing Inference"):
                    input_ids = batch["gene_ids"].to(self.task.args.device)
                    input_vals = batch["values"].to(self.task.args.device)
                    batch_ids = batch["batch_labels"].to(self.task.args.device)
                    pad_mask = input_ids.eq(self.task.vocab[self.task.pad_token])
                    out = self.model._encode(src=input_ids, values=input_vals, batch_labels=batch_ids,
                                             src_key_padding_mask=pad_mask)
                    batch_emb = self.model._get_cell_emb_from_layer(out, input_vals,
                                                                    src_key_padding_mask=pad_mask).cpu().numpy()

                    all_batch_labels.append(batch_ids.cpu().numpy())
                    all_celltype_labels.append(batch["celltype_labels"].cpu().numpy())
                    embeddings[count:count + len(batch_emb)] = batch_emb
                    count += len(batch_emb)

            # Save state for subsequent steps
            type(self).raw_embeddings = embeddings
            type(self).batch_labels = np.concatenate(all_batch_labels)
            type(self).celltype_labels = np.concatenate(all_celltype_labels)

            self.assertEqual(embeddings.shape[0], len(self.loader.dataset))
            self.logger.info("--- Emb [Step 2/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Emb [Step 2/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 2 failed: {e}")

    def test_step_03_refine_and_save_embeddings(self):
        """Step 3: Verify the embedding post-processing and saving functionality."""
        self.logger.info("--- Emb [Step 3/4]: Testing embedding refinement and saving ---")
        try:
            self.assertTrue(hasattr(self, 'raw_embeddings'), "Raw embeddings from step 2 are missing.")

            # Replicate the post-processing and saving logic from run_embedding()
            embeddings = refine_embedding(self.raw_embeddings, self.batch_labels, self.celltype_labels, 2)
            save_path = self.task.save_dir / "cell_embedding.npy"
            np.save(save_path, embeddings)

            # Save state for the next step
            type(self).final_embeddings = embeddings

            self.assertTrue(save_path.exists())
            self.logger.info("--- Emb [Step 3/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Emb [Step 3/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 3 failed: {e}")

    def test_step_04_organize_results(self):
        """Step 4: Verify the final result organization and visualization functionality."""
        self.logger.info("--- Emb [Step 4/4]: Testing result organization and visualization ---")
        try:
            self.assertTrue(hasattr(self, 'final_embeddings'), "Final embeddings from step 3 are missing.")

            # Load the original adata to pass to _organize_results
            adata = sc.read_h5ad(self.task.args.data_path)
            self.task._organize_results(adata, self.final_embeddings)

            # Check if the UMAP plot was generated, based on the config
            if getattr(self.task.args, "draw_umap", False):
                umap_file = self.task.save_dir / "cell_emb_umap.png"
                self.assertTrue(umap_file.exists())

            self.logger.info("--- Emb [Step 4/4]: PASSED ✅ ---")
        except Exception as e:
            self.logger.error(f"Emb [Step 4/4]: FAILED with exception: {e}", exc_info=True)
            self.fail(f"Step 4 failed: {e}")


if __name__ == '__main__':
    # Redirect all unittest output (including tracebacks) to the log file
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestEmbTask))
        runner = unittest.TextTestRunner(verbosity=2, stream=log_file)
        print(f"Running tests for Embedding task. See detailed logs in {LOG_FILE_PATH}")
        runner.run(suite)