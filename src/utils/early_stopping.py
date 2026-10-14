import numpy as np


class EarlyStopping:
    """
    Early stopping basado en el MAE dentro de la region de la lesion (lesion_mae, menor es mejor).

    Por que el cambio: con SSIM/MAE globales (sobre el slice completo), la lesion tipicamente
    representa <0.3% de los pixeles, por lo que el modelo puede obtener un score global excelente
    incluso si borra o alucina la lesion por completo. lesion_mae se calcula SOLO dentro de la
    mascara de lesion del paciente, asi que mide directamente lo que se quiere preservar.

    ssim y mae globales se siguen recibiendo y guardando (para contexto/desempate visual en logs),
    pero NO deciden si se guarda el checkpoint.

    El guardado del modelo lo maneja el Trainer, no esta clase.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_lesion_mae = np.inf
        self.best_ssim = -np.inf
        self.best_mae = np.inf
        self.counter = 0
        self.best_step = 0
        self.stop_training = False

    def __call__(self, lesion_mae: float, ssim: float, mae: float, step: int) -> bool:
        improved = float(lesion_mae) < self.best_lesion_mae - self.min_delta

        if improved:
            if self.verbose:
                print(f"[EarlyStopping] Step {step}: lesion_mae {self.best_lesion_mae:.6f} -> {lesion_mae:.6f} "
                      f"| ssim_global={ssim:.6f} mae_global={mae:.6f}. Guardando checkpoint...")
            self.best_lesion_mae = lesion_mae
            self.best_ssim = ssim
            self.best_mae = mae
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Step {step}: lesion_mae={lesion_mae:.6f} (mejor={self.best_lesion_mae:.6f}) "
                      f"| ssim_global={ssim:.6f} mae_global={mae:.6f}. Sin mejora {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"[EarlyStopping] Paciencia agotada. Mejor step: {self.best_step} "
                          f"(lesion_mae={self.best_lesion_mae:.6f}, ssim_global={self.best_ssim:.6f}, "
                          f"mae_global={self.best_mae:.6f}).")
                self.stop_training = True

        return self.stop_training
