import unittest
import os
import shutil
import toml
import sys
import logging
from pathlib import Path

# Add the project root to the Python path to ensure modules can be imported
sys.path.append('/home/share/huadjyin/home/s_huluni/gzh/RegFormer')
from downstream_task.regformer_grn import GrnTaskMamba
import networkx as nx

# --- Logger Configuration ---
LOG_FILE_PATH = "/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_grn/grn_test.log"
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


class TestGRNTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the testing environment once for all tests in this class.
        1. Load the real configuration for the GRN task.
        2. Create a safe, temporary output directory.
        3. Initialize the GrnTaskMamba instance for testing.
        """
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=" * 20 + " Starting GRN Task Test Suite " + "=" * 20)

        # --- 1. Load the real 'grn.toml' config file ---
        real_config_filename = "grn_10k.toml"
        real_config_path = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/configs") / real_config_filename
        if not real_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {real_config_path}")
        config = toml.load(real_config_path)

        # --- 2. [Safety Measure] Create a temporary output directory ---
        cls.test_dir = Path("/home/share/huadjyin/home/s_huluni/gzh/RegFormer/Docs/test/RegFormer_grn/temp_test_output_grn")
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(exist_ok=True)
        config['save_dir'] = str(cls.test_dir)
        # The GRN task does not have epochs, but this is good practice
        config['epochs'] = 1

        safe_config_path = cls.test_dir / "safe_config_for_test.toml"
        with open(safe_config_path, "w") as f:
            toml.dump(config, f)

        # --- 3. Initialize the Task instance ---
        cls.logger.info("STEP 0: Initializing GrnTaskMamba instance...")
        cls.task = GrnTaskMamba(config_file=safe_config_path)
        cls.logger.info("STEP 0: Initialization complete.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources once after all tests in this class have run.
        This includes closing loggers and removing the temporary directory.
        """
        cls.logger.info("Cleaning up resources for GRN Task...")

        # --- Step 2: Close and remove custom file log handlers ---
        if hasattr(cls.task, 'logger'):
            cls.logger.info("Closing custom file logger to release 'run.log'...")
            for handler in cls.task.logger.handlers[:]:
                handler.close()
                cls.task.logger.removeHandler(handler)

        # --- Step 3: Remove the temporary directory ---
        cls.logger.info("Cleaning up temporary directory...")
        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir)
                cls.logger.info("Temporary directory successfully removed.")
            except OSError as e:
                cls.logger.error(f"Failed to remove temporary directory: {e}. It might need to be manually deleted.")

        cls.logger.info("=" * 20 + " GRN Task Test Suite Finished " + "=" * 20)

    def test_step_01_get_gene_embeddings(self):
        """Step 1: Verify that gene embeddings can be generated and saved successfully."""
        self.logger.info("--- GRN [Step 1/3]: Testing gene embedding generation ---")
        # This corresponds to the first major action in run_grn_analysis()
        embeddings = self.task.get_gene_expression_embedding()

        # Save the result for the next test step
        type(self).embeddings = embeddings

        # Assertions to verify the outcome of this step
        self.assertIsInstance(self.embeddings, dict, "Embeddings should be a dictionary.")
        self.assertGreater(len(self.embeddings), 0, "Embeddings dictionary should not be empty.")
        self.assertTrue((self.task.save_dir / "gene_embedding.npy").exists(), "gene_embedding.npy was not created.")
        self.logger.info("--- GRN [Step 1/3]: PASSED ✅ ---")

    def test_step_02_construct_grn(self):
        """Step 2: Verify that the GRN graph can be constructed from the embeddings."""
        self.logger.info("--- GRN [Step 2/3]: Testing GRN construction ---")
        # This corresponds to the second major action in run_grn_analysis()
        # It depends on the 'embeddings' generated in the previous step
        self.assertTrue(hasattr(self, 'embeddings'), "Embeddings from previous step are missing.")
        grn_graph = self.task.construct_grn(self.embeddings)

        # Save the result for the next test step
        type(self).grn_graph = grn_graph

        # To faithfully test the pipeline, we also save the edges like in the main script
        edges_df = nx.to_pandas_edgelist(grn_graph)
        edges_df.to_csv(self.task.save_dir / "edges.csv", index=False)

        # Assertions to verify the outcome of this step
        self.assertIsInstance(self.grn_graph, nx.Graph, "Output should be a networkx Graph object.")
        self.assertGreater(self.grn_graph.number_of_nodes(), 0, "Graph should have nodes.")
        self.assertTrue((self.task.save_dir / "edges.csv").exists(), "edges.csv was not created.")
        self.logger.info("--- GRN [Step 2/3]: PASSED ✅ ---")

    def test_step_03_evaluate_grn(self):
        """Step 3: Verify that the constructed GRN can be evaluated."""
        self.logger.info("--- GRN [Step 3/3]: Testing GRN evaluation ---")
        # This corresponds to the final action in run_grn_analysis()
        # It depends on the 'grn_graph' created in the previous step
        self.assertTrue(hasattr(self, 'grn_graph'), "GRN graph from previous step is missing.")
        self.task.evaluate_grn(self.grn_graph)

        # Assertion to verify the final output of the pipeline
        results_file = self.task.save_dir / "grn_enrichment.csv"
        self.assertTrue(results_file.exists(), "grn_enrichment.csv was not created.")
        self.logger.info("--- GRN [Step 3/3]: PASSED ✅ ---")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGRNTask))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)