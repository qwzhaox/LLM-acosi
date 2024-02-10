#!/bin/bash

bash scripts/bash/run.sh --task "acos-extract" --model "gpt-4-long"
bash scripts/bash/run.sh --task "acosi-extract" --model "gpt-4-long"
bash scripts/bash/run.sh --task "acos-extend" --model "gpt-4-long"

bash scripts/bash/run.sh --task "acos-extract" --model "gpt-35-long"
bash scripts/bash/run.sh --task "acosi-extract" --model "gpt-35-long"
bash scripts/bash/run.sh --task "acos-extend" --model "gpt-35-long"