import logging

import torch
from data_utils.torch_dataset import get_train_test_datasets
from torch import nn
from tqdm import tqdm

from .settings import TrainingSettings

logger = logging.getLogger(__name__)


def run_training(model: nn.Module, cfg: TrainingSettings) -> nn.Module:
    """
    Run the model training loop and return the trained model
    """
    cfg.save_to.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting training run. Checkpoints will be saved to %s", cfg.save_to
    )
    model.train()

    # Prepare data loaders
    train_set, test_set = get_train_test_datasets(
        cfg.dataset_path,
        image_size=cfg.image_size,
        perturb_cfg=cfg.data_perturbation,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=cfg.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=cfg.batch_size
    )

    # Prepare optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(cfg.epochs):
        logger.info("Starting epoch (%d / %d)", epoch + 1, cfg.epochs)
        run_epoch(model, train_loader, optimiser)

        logger.info("Starting evaluation run")
        metrics = run_evaluation(model, test_loader)
        logger.info(
            "Evaluation done! Avg Validation Loss=%.5f - Accuracy=%.4f",
            *metrics,
        )

    torch.save(
        model.state_dict(), cfg.save_to / f"{type(model).__name__}-final.pth"
    )
    return model


def run_epoch(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
):
    """
    Run a full epoch on the training data.
    """
    accum_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    batch = 0
    for x, y in train_data:
        batch += 1
        optimiser.zero_grad()

        logits = model(x)  # Shape (batch, 10, 4)
        loss = loss_fn(logits, y)

        # Optimiser step
        loss.backward()
        optimiser.step()
        loss_num = loss.item()
        accum_loss += loss_num

        if batch % 100 == 0:
            logger.info(
                "Ran batch %03d of epoch. Batch loss: %.5f", batch, loss_num
            )

    avg_loss = accum_loss / batch
    logger.info(
        "Epoch complete. Ran %03d batches. Average Train Loss: %.5f",
        batch,
        avg_loss,
    )
    return model


def run_evaluation(
    model: nn.Module,
    eval_data: torch.utils.data.DataLoader,
    eval_mode: bool = True,
) -> tuple[float, float]:
    """
    Run model evaluation on the given data loader. Return the accuracy and the average loss.
    """
    total_correct = 0
    total_examples = 0
    batch = 0
    loss_accum = 0.0
    loss_fn = nn.CrossEntropyLoss()

    if eval_mode:
        model.eval()

    with torch.no_grad():
        for x, y in tqdm(eval_data):
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_accum += loss.item()

            labels = torch.argmax(logits, dim=1).to(torch.long)
            correct = torch.sum(torch.where(labels == y, 1, 0)).item()
            total_correct += correct
            total_examples += y.shape[0] * 4

    if eval_mode:
        model.train()
    return loss_accum / max(batch, 1), total_correct / max(total_examples, 1)
