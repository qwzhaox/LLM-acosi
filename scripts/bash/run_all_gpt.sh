#!/bin/bash

bash scripts/bash/run.sh --task "acosi-extract" --model "gpt-35-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extend" --model "gpt-35-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-35-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-35-long" --dataset rest
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-35-long" --dataset laptop

bash scripts/bash/run.sh --task "acosi-extract" --model "gpt-4-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extend" --model "gpt-4-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-4-long" --dataset shoes
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-4-long" --dataset rest
bash scripts/bash/run.sh --task "acos-extract" --model "gpt-4-long" --dataset laptop