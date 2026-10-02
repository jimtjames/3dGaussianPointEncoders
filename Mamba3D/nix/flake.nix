{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable-small";
    flake-utils.url = "github:numtide/flake-utils";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];

    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, nix-gl-host}:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        cudatookit-with-cudart-to-lib64 = pkgs.symlinkJoin {
          name = "cudatoolkit";
          paths = with pkgs.cudaPackages; [
            cudatoolkit
            #cuda_nvcc
            #(pkgs.lib.getLib cuda_nvcc)
            #(pkgs.lib.getDev cuda_nvcc)
            #(pkgs.lib.getLib cuda_nvrtc)
            #(pkgs.lib.getDev cuda_nvrtc)
            #(pkgs.lib.getLib cuda_cudart)
            #(pkgs.lib.getDev cuda_cudart)
            (pkgs.lib.getStatic cuda_cudart)
            #(pkgs.lib.getStatic cuda_nvrtc)
            #(pkgs.lib.getStatic cuda_nvrtc)
          ];
          postBuild = ''
            ln -s $out/lib $out/lib64
          '';
        };
        #pkgs-mmcv = import nixpkgs-mmcv {
          #inherit system;
          #config = {
            #allowUnfree = true;
            #cudaSupport = true;
          #};
        #};
      in {
        # https://nixos.org/manual/nixpkgs/stable/#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems
        devShells.default =
          let
            # Define the list of required system libraries once
            systemLibs = with pkgs; [
              xorg.libX11
              xorg.libXext
              xorg.libXfixes
              xorg.libXrender
              libglvnd # Provides OpenGL libraries (libGL.so)
              udev
              gcc
              libgcc.lib
              #stdenv.cc.lib
              #libgomp
            ];
          in
          pkgs.mkShell {
          name = "impurePythonEnv";
          venvDir = "./.venv";
          nativeBuildInputs = systemLibs;

          buildInputs = with pkgs; [
            # A Python interpreter including the 'venv' module is required to bootstrap
            # the environment.
            python313Packages.python

            # This executes some shell code to initialize a venv in $venvDir before
            # dropping into the shell
            python313Packages.venvShellHook

            # Those are dependencies that we would like to use from nix which will
            # add them to PYTHONPATH and thus make them accessible from within the venv.
            #python313Packages.mmengine
            python313Packages.ultralytics-thop
            python313Packages.mmcv
            python313Packages.numpy
            python313Packages.torch
            python313Packages.numbaWithCuda
            python313Packages.easydict
            python313Packages.timm
            python313Packages.h5py
            python313Packages.matplotlib
            python313Packages.pyyaml
            python313Packages.scipy
            python313Packages.scikit-learn
            python313Packages.tensorboardx
            python313Packages.tqdm
            python313Packages.termcolor
            python313Packages.transforms3d
            python313Packages.ninja
            python313Packages.opencv-python
            python313Packages.fvcore
            python313Packages.opentsne
            python313Packages.optuna
            python313Packages.plotly
            #python313Packages.ray

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
            #ninja
            basedpyright
            jupyter-all
            nodejs_22
            (import nix-gl-host { inherit pkgs; })

          ];

          # Run this command, only after creating the virtual environment
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
            export CUDA_PATH=${cudatookit-with-cudart-to-lib64}
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (systemLibs ++ [ cudatookit-with-cudart-to-lib64 ])}''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            nixglhost pip install -r requirements.txt
            #mim install "mmcv==2.1.0"
            #mim install "mmdet==3.3.0"
            #mim install "mmdet3d==1.4.0"
          '';

          #shellHook = ''
            #export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemLibs}''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"

            ## Allow pip to install wheels (from your original config).
            #unset SOURCE_DATE_EPOCH
          #'';

          # Now we can execute any commands within the virtual environment.
          # This is optional and can be left out to run pip manually.
          postShellHook = ''
            export CUDA_PATH=${cudatookit-with-cudart-to-lib64}
            # Add the CUDA toolkit to the library path
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (systemLibs ++ [ cudatookit-with-cudart-to-lib64 ])}''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            # allow pip to install wheels
            unset SOURCE_DATE_EPOCH
          '';
          #postShellHook = ''
            #export CUDA_PATH=${cudatookit-with-cudart-to-lib64}
            #export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemLibs}''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
            ## allow pip to install wheels
            #unset SOURCE_DATE_EPOCH
          #'';
        };
      }
    );
}

