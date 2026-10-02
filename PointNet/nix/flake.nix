{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable-small";
    flake-utils.url = "github:numtide/flake-utils";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];

    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, nix-gl-host }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            #cudaCapabilities = [ "8.6" "8.9" ];
          };
        };
        #python = pkgs.python311;
      in {
        # https://nixos.org/manual/nixpkgs/stable/#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems
        devShells.default = pkgs.mkShell {
          name = "impurePythonEnv";
          venvDir = "./.venvNew";
          buildInputs = with pkgs; [
            # A Python interpreter including the 'venv' module is required to bootstrap
            # the environment.
            python3Packages.python

            # This executes some shell code to initialize a venv in $venvDir before
            # dropping into the shell
            python3Packages.venvShellHook

            # Those are dependencies that we would like to use from nixpkgs, which will
            # add them to PYTHONPATH and thus make them accessible from within the venv.
            python3Packages.tqdm
            #python3Packages.jupyter
            python3Packages.h5py
            python3Packages.numpy
            python3Packages.torch
            python3Packages.torchvision
            python3Packages.triton
            python3Packages.matplotlib
            python3Packages.scipy
            python3Packages.plotly
            python3Packages.pandas

            python3Packages.h5py
            python3Packages.ray
            #python3Packages.optuna
            #python3Packages.gpy

            (import nix-gl-host { inherit pkgs; })

            # In this particular example, in order to compile any binary extensions they may
            # require, the Python modules listed in the hypothetical requirements.txt need
            # the following packages to be installed locally:
            # taglib
            # openssl
            # git
            # libxml2
            # libxslt
            # libzip
            # zlib
            jupyter-all
            pprof
          ];

          # Run this command, only after creating the virtual environment
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
            pip install -r requirements.txt
          '';

          # Now we can execute any commands within the virtual environment.
          # This is optional and can be left out to run pip manually.
          postShellHook = ''
            # allow pip to install wheels
            unset SOURCE_DATE_EPOCH
            export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib
          '';
        };
      }
    );
}
