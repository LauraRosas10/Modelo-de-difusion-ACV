# PASTA-3D (NCCT -> DWI, ACV)

Version 3D del repo `Modelo-de-difusion-ACV` (antes 2.5D: cortes vecinos como canales).

## Uso
```bash
# 0) datos: los .h5 de prepare_pasta_h5.py se usan SIN cambios (ya guardan volumenes 3D completos)
# 1) prueba rapida del pipeline en CPU (datos sinteticos)
python tests/make_synthetic_h5.py /tmp/syn/
python tests/test_3d_pipeline.py
python train_mri2pet.py --config src/config/pasta3d_smoke_test.yaml
# 2) entrenar
python train_mri2pet.py --config src/config/pasta3d_acv_train.yaml
# 3) evaluar / sintetizar (usar el MISMO timestep_respacing y arquitectura del entrenamiento)
python train_mri2pet.py --config src/config/pasta3d_acv_eval.yaml
```
Ajusta `data_dir` y `results_folder` en los YAML. Si hay OOM: baja `patch_size`, `unet_dim`, `num_res_blocks`,
pon `with_attention: false`, o deja `use_checkpoint: true` y `cycle_detach: true`.
Configs 2.5D antiguas: `src/config/legacy_2_5d/` (no son compatibles con este codigo; usa la rama/tag anterior).
Explicacion detallada de cambios: ver el documento entregado en el chat.
