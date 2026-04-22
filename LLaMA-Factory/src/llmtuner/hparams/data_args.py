from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(#数据集名称
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    eval_dataset_dir: str = field(
        default="data",
    )
    output_result_dir: str = field(
        default="result",
    )
    use_prompt: bool = field(
        default=False,
    )
    target: str = field(
        default="data",
    )
    
    #new mode
    selected_anchor: Optional[str] = field(
        default=None,
        metadata={"help": "Optional anchor entity for ROCR step2 redirection. If not set, use default target."},
    )

    '''
    rocr_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Override mode used by ROCR (e.g. reject, noise, unrelated, embed_proj)."},
    )
    anchor_selection_method: str = field(
        default="none",
        metadata={"help": "Anchor selection method: none/random/most_similar/strongest/adaptive."},
    )
    anchor_candidate_names: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated candidate anchor entities. If None, infer from eval_dataset_dir."},
    )
    anchor_position_mode: str = field(
        default="last_token",
        metadata={"help": "Position mode for representation extraction: last_token or span_pool."},
    )
    anchor_template_type: str = field(
        default="default",
        metadata={"help": "Template preset for anchor feature extraction."},
    )
    anchor_top_m: int = field(
        default=5,
        metadata={"help": "Top-M anchors kept after static ranking."},
    )
    anchor_use_proxy: bool = field(
        default=True,
        metadata={"help": "Whether to run proxy reranking for top-M anchors."},
    )
    anchor_mu: float = field(
        default=0.5,
        metadata={"help": "SimBand center mu."},
    )
    anchor_alpha_var: float = field(
        default=1.0,
        metadata={"help": "Variance penalty coefficient in strength score."},
    )
    anchor_beta_retain: float = field(
        default=1.0,
        metadata={"help": "beta for retain-drop in proxy score."},
    )
    anchor_gamma_collapse: float = field(
        default=1.0,
        metadata={"help": "gamma for collapse penalty in proxy score."},
    )
    anchor_lambda_proxy: float = field(
        default=1.0,
        metadata={"help": "lambda for combining static and proxy scores."},
    )
    anchor_ablation: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated ablations: no_sim_band,no_strength,no_safe,no_compat,no_proxy."},
    )
    '''
    split: str = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    cutoff_len: int = field(
        default=1024,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    reserved_label_len: int = field(
        default=1,
        metadata={"help": "The minimum cutoff length reserved for the tokenized labels in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."
        },
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether or not to pack the sequences in training. Will automatically enable in pre-training."
        },
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the tokenized datasets."},
    )

    def __post_init__(self):
        if self.reserved_label_len >= self.cutoff_len:
            raise ValueError("`reserved_label_len` must be smaller than `cutoff_len`.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")
