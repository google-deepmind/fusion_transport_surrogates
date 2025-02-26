# Fusion transport surrogates

A library of surrogate transport models for tokamak fusion.

This library provides both inference code and model weights and metadata. It is
designed to provide surrogate transport models for
[TORAX](https://github.com/google-deepmind/torax), but
the models should be usable by other fusion simulators.


## Installation instructions

Install virtualenv (if not already installed):

```shell
pip install --upgrade pip
```

```shell
pip install virtualenv
```

Create a code directory where you will install the virtual env and other
fusion_transport_surrogates dependencies.

```shell
mkdir /path/to/fusion_transport_surrogates_dir && cd "$_"
```
Where `/path/to/fusion_transport_surrogates_dir` should be replaced by a path
of your choice.


Create a fusion_transport_suurrogates virtual env:

```shell
python3 -m venv fusion_transport_surrogates_venv
```

Activate the virtual env:

```shell
source fusion_transport_surrogates_venv/bin/activate
```

Download and install the library via http:

```shell
git clone https://github.com/google-deepmind/fusion_transport_surrogates.git
```
or ssh (ensure that you have the appropriate SSH key uploaded to github).

```shell
git clone git@github.com:google-deepmind/fusion_transport_surrogates.git
```


Enter the fusion_transport_surrogates directory

```shell
cd fusion_transport_surrogates
```

Install the library:

```shell
pip install -e .
```

If you want to run unit tests, install with the `testing` option:

```shell
pip install -e .[testing]
pytest
```

## Disclaimer
Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license
at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
