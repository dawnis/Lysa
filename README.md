# Lysa
Python codebase to create Lysa using Evo RL

## Evo RL Installation Instructions
In order to install the latest version of Evo RL, follow these steps:
1. activate the package environment, (in this case, Lysa): `conda activate lysa`
2. Navigate to the Rust package for Evo RL on your machine. 
3. Run `maturin develop` which should install the latest version into your environment. 

## Troubleshooting
* If a new version is released, `pip` may not recognize this and automatically replace it with the most updated one. To troubleshoot this situation, simply run `pip uninstall evo_rl` and then re-install using the procedure above. 