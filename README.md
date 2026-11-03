# Chibi — 8K remaster

Единая стилизованная модель, собранная из тела из Blender и отдельной
STL-головы, после чего полностью пересобранная в высокодетализированную
версию с кинематографичной текстурой кожи.

## Главная версия: `Chibi_pretty`

- **`Chibi_pretty.glb`** — рекомендуемая готовая модель. Один mesh,
  418 828 вершин и 837 652 треугольника; все три карты имеют размер
  8192×8192 и уже встроены в файл.
- `Chibi_pretty.blend` — та же модель для Blender: один mesh-объект,
  UV-развёртка, слой вершинной росписи `SkinPaint` и все три PNG-мастера,
  упакованные внутрь файла без потерь.
- `Chibi_pretty.stl` — только геометрия для печати и других 3D-инструментов:
  одна замкнутая watertight-поверхность, один компонент, без граничных
  рёбер. Формат STL не хранит цвет и текстуры.
- `textures/skin_basecolor.png` — базовая карта кожи 8192×8192: мягкий
  градиент, розовые щёчки, носик, губы и веки, веснушки через носик и щёчки
  и нежные розовые кончики пальцев.
- `textures/skin_roughness.png` — карта шероховатости 8192×8192.
- `textures/skin_normal.png` — tangent-space normal-карта 8192×8192 с
  широким мягким рельефом кожи и мелкими порами.
- `Chibi_pretty_report.json` — параметры сборки, росписи, запекания и
  геометрии.
- Cycles-превью 1000×1333: [спереди](preview/Chibi_pretty_front.png),
  [в три четверти](preview/Chibi_pretty_threequarter.png) и
  [крупный план](preview/Chibi_pretty_closeup.png).

![Chibi pretty front preview](preview/Chibi_pretty_front.png)

![Chibi pretty close-up](preview/Chibi_pretty_closeup.png)

### Как устроен ремастер

Красивая версия строится напрямую из `Chibi_full.stl` с полной исходной
детализацией. Используются voxel-remesh `0.12` и деликатное сглаживание
`0.25 × 2`, поэтому мелкие исходные формы равномерно сохраняются по всей
поверхности. Габариты модели — 46.6228 × 19.0937 × 76.7788, знаковый объём —
7871.0351, площадь поверхности — 4062.0471.

Все карты запекаются строго в 8192×8192 — без уменьшенных 4K/2K-копий.
В GLB базовая карта хранится как JPEG качества 92 в полном разрешении
8192×8192, а служебные roughness- и normal-карты остаются PNG. Это нужно,
чтобы самодостаточный GLB помещался в лимит GitHub 100 МБ на файл.
Lossless PNG-мастера лежат в папке `textures/` и упакованы в
`Chibi_pretty.blend`.

Актуальные размеры основных файлов:

| Файл | Размер |
| --- | ---: |
| `Chibi_pretty.glb` | 85 321 680 байт |
| `Chibi_pretty.blend` | 87 695 664 байт |
| `Chibi_pretty.stl` | 41 882 684 байт |
| `textures/skin_basecolor.png` | 21 095 180 байт |
| `textures/skin_roughness.png` | 9 676 696 байт |
| `textures/skin_normal.png` | 34 914 299 байт |

## Исходники и объединённая база

- `body base.blend` — исходное тело с отдельной шеей и арматурой.
- `Chibi_head.stl` — исходная голова в формате binary STL.
- `Chibi_full.stl` — объединённая модель с полной исходной детализацией;
  именно она используется как вход для 8K-ремастера.
- `Chibi_full.blend` — объединённый mesh-объект в Blender.
- `Chibi_full.glb` — объединённая модель без финальной текстуры кожи.
- `Chibi_full_report.json` — параметры и геометрия базовой сборки.

Старый промежуточный `Chibi_full_fused.stl` и старые ортографические листы
превью удалены из текущего дерева: актуальная высокодетализированная
watertight-геометрия теперь находится в `Chibi_pretty.stl`.

## Пересборка в GitHub Actions

### Базовая объединённая модель

1. Откройте вкладку **Actions**.
2. Выберите **Build unified Chibi model**.
3. Нажмите **Run workflow**.
4. При необходимости измените параметры:
   - `head_scale` — масштаб головы, по умолчанию `0.60`;
   - `voxel_size` — размер вокселя для промежуточной слитой STL,
     по умолчанию `0.30`;
   - `commit_results` — записывать ли готовые файлы обратно в ветку.

### 8K-ремастер с текстурой кожи

1. Откройте вкладку **Actions**.
2. Выберите **Build pretty textured Chibi model**.
3. Нажмите **Run workflow**.
4. При необходимости измените параметры:
   - `voxel_size` — плотность поверхности, по умолчанию `0.12`;
   - `smooth_factor` — сила сглаживания, по умолчанию `0.25`;
   - `smooth_iterations` — число итераций сглаживания, по умолчанию `2`;
   - `texture_size` — размер каждой карты кожи, по умолчанию `8192`;
   - `bake_samples` — Cycles-сэмплы при запекании, по умолчанию `8`;
   - `render_samples` — Cycles-сэмплы превью, по умолчанию `64`;
   - `render_width` — ширина превью, по умолчанию `1000`;
   - `commit_results` — записывать ли готовые файлы обратно в ветку.

Workflow запускает `scripts/enhance_chibi.py`, записывает PNG-мастера
энкодером максимальной компрессии, переупаковывает базовую карту внутри GLB
скриптом `scripts/repack_glb_basecolor.py`, проверяет размеры и содержимое
всех карт, валидирует STL/GLB и создаёт три превью. Полная 8K-сборка обычно
занимает около 35–45 минут.

## Локальная сборка

### Объединение тела и головы

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

### 8K-ремастер

После создания объединённой `Chibi_full.stl` выполните:

```bash
blender --background --factory-startup -noaudio \
  --python scripts/enhance_chibi.py -- \
  --input Chibi_full.stl \
  --out-dir . \
  --voxel-size 0.12 \
  --smooth-factor 0.25 \
  --smooth-iterations 2 \
  --texture-size 8192 \
  --bake-samples 8 \
  --render-samples 64 \
  --render-width 1000
```

Для финальной переупаковки базовой карты внутри GLB нужен Python с Pillow:

```bash
python3 scripts/repack_glb_basecolor.py Chibi_pretty.glb --quality 92
```

Проверка геометрии:

```bash
python3 scripts/validate_stl.py Chibi_pretty.stl \
  --require-watertight \
  --require-single-component \
  --require-positive-volume \
  --min-triangles 700000
```

> В готовых моделях поза тела «запечена»: это статичные модели без арматуры и
> vertex groups. Исходный риг по-прежнему находится в `body base.blend`.
