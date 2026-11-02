# Chibi

Единая модель, собранная из тела из Blender и отдельной STL-головы.

## Исходные файлы

- `body base.blend` — тело с отдельной шеей и арматурой.
- `Chibi_head.stl` — голова в формате binary STL.

## Готовая модель

- **`Chibi_full_fused.stl`** — основная единая модель: одна замкнутая
  watertight-поверхность после voxel-remesh. Этот файл удобнее всего
  использовать как цельную модель, в том числе для 3D-печати.
- `Chibi_full.blend` — один объединённый mesh-объект в Blender с сохранённой
  исходной геометрией тела и головы.
- `Chibi_full.stl` — STL объединённого mesh-объекта без voxel-remesh.
- `Chibi_full.glb` — GLB-версия объединённой модели для просмотрщиков.
- `Chibi_full_report.json` — размеры, количество вершин/полигонов и параметры
  сборки.
- `preview/Chibi_full_fused_views.png` — виды спереди, сзади, сбоку, сверху и
  крупный план головы/шеи.

![Chibi full fused preview](preview/Chibi_full_fused_views.png)

## Как пересобрать в GitHub Actions

1. Откройте вкладку **Actions**.
2. Выберите **Build unified Chibi model**.
3. Нажмите **Run workflow**.
4. При необходимости измените параметры:
   - `head_scale` — масштаб головы, по умолчанию `0.60`;
   - `voxel_size` — размер вокселя для слитой STL, по умолчанию `0.30`;
   - `commit_results` — записывать ли готовые файлы обратно в выбранную ветку.

Workflow скачивает официальный Blender 4.2.0, запускает
`scripts/merge_chibi.py`, проверяет STL и создаёт превью. Все файлы также
доступны как artifact завершённого запуска.

## Сборка локально

```bash
blender --background --factory-startup -noaudio \
  --python scripts/merge_chibi.py -- \
  --body "body base.blend" \
  --head Chibi_head.stl \
  --out-dir . \
  --head-scale 0.60 \
  --voxel-size 0.30
```

Голова в исходном STL смотрит в противоположную телу сторону, поэтому при
сборке она поворачивается на 180° вокруг оси Z и ставится на ось шеи с
небольшим перекрытием, чтобы на стыке не было щели.

> В готовой модели поза тела «запечена»: это статичная модель без арматуры и
> vertex groups. Исходный риг по-прежнему находится в `body base.blend`.
