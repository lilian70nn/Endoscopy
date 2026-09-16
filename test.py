import torch
from PIL import Image

from cortex_dataloader import PrefetchDataLoader, CortexDataLoader, NUM_IMAGES
from dino import DataAugmentationDINO, DINOHead, DINOLoss, cosine_schedule as dino_schedule
from lejepa import DataAugmentationLeJEPA, sigreg, cosine_schedule as lejepa_schedule


def test_steps_per_epoch():
    class FakeLoader:
        batch_size = 256

    loader = PrefetchDataLoader(FakeLoader(), prefetch_batches=2, devices=4)

    assert len(loader) == NUM_IMAGES // (256 * 4)
    assert len(loader) == 4707
    assert loader.batch_size == 256

    print("[PASS] steps_per_epoch =", len(loader))


def test_dino_augmentation():
    image = Image.new("RGB", (640, 480))

    aug = DataAugmentationDINO(local_crops_number=6)
    crops = aug(image)

    assert len(crops) == 8
    assert crops[0].shape == (3, 224, 224)
    assert crops[1].shape == (3, 224, 224)

    for crop in crops[2:]:
        assert crop.shape == (3, 96, 96)

    print("[PASS] DINO augmentation")


def test_lejepa_augmentation():
    image = Image.new("RGB", (640, 480))

    aug = DataAugmentationLeJEPA(local_crops_number=6)
    crops = aug(image)

    assert len(crops) == 8
    assert crops[0].shape == (3, 224, 224)
    assert crops[1].shape == (3, 224, 224)

    for crop in crops[2:]:
        assert crop.shape == (3, 96, 96)

    print("[PASS] LeJEPA augmentation")


def test_dino_head():
    head = DINOHead(
        in_dim=768,
        out_dim=8000,
        hidden_dim=1600,
        bottleneck_dim=256,
    )

    x = torch.randn(4, 768)
    y = head(x)

    assert y.shape == (4, 8000)

    print("[PASS] DINO head")


def test_dino_loss():
    cfg = {
        "student_temp": 0.1,
        "center_momentum": 0.9,
        "local_crops_number": 6,
        "out_dim": 8000,
        "warmup_teacher_temp_epochs": 0,
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.04,
        "epochs": 5,
    }

    loss_fn = DINOLoss(cfg)

    batch_size = 2
    student = torch.randn(batch_size * 8, 8000, requires_grad=True)
    teacher = torch.randn(batch_size * 2, 8000)

    loss = loss_fn(student, teacher, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert student.grad is not None

    print("[PASS] DINO loss + backward")


def test_sigreg():
    x = torch.randn(8, 768, requires_grad=True)

    loss = sigreg(x, global_step=0, num_slices=16).mean()

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("[PASS] LeJEPA SIGReg + backward")


def test_schedules():
    steps = 4707
    epochs = 5

    dino_lr = dino_schedule(
        5e-4,
        1e-6,
        epochs * steps,
        steps,
    )

    lejepa_lr = lejepa_schedule(
        5e-4,
        1e-6,
        epochs * steps,
        steps,
    )

    assert len(dino_lr) == epochs * steps
    assert len(lejepa_lr) == epochs * steps

    print("[PASS] schedules =", len(dino_lr))


if __name__ == "__main__":
    print("Running CPU tests...\n")

    test_steps_per_epoch()
    test_dino_augmentation()
    test_lejepa_augmentation()
    test_dino_head()
    test_dino_loss()
    test_sigreg()
    test_schedules()

    print("\nALL CPU TESTS PASSED")