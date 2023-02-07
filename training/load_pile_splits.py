# Copyright 2020 The HuggingFace Datasets Authors and
# the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import gzip

import datasets
from dataclasses import dataclass


# TODO: all this stuff
_CITATION = """Pile Subsets"""
_DESCRIPTION = """Pile Subsets"""
_HOMEPAGE = ""
_LICENSE = ""


@dataclass
class PileSubsetConfig(datasets.BuilderConfig):
    name = "pile_subset"
    filename: str = "pile_subset.csv"  # Required!


class PileSubset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.5.0")

    BUILDER_CONFIG_CLASS = PileSubsetConfig

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "meta": datasets.Sequence({
                    "pile_set_name": datasets.Value("string"),
                })
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, _):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": self.config.filename,
                },
            ),
        ]

    def _generate_examples(self, filepath):
        with gzip.open(filepath) as f:
            for key, row in enumerate(f):
                if row.strip() == b'':
                    continue
                yield key, json.loads(row)
