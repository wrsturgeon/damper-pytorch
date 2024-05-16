{
  description = "Neural network optimizer based solely on reducing oscillation.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixfmt = {
      inputs.flake-utils.follows = "flake-utils";
      url = "github:serokell/nixfmt";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      flake-utils,
      nixfmt,
      nixpkgs,
      self,
    }:
    let
      pname = "damper-pytorch";
      pyname = "damper_pytorch";
      version = "0.0.1";
      src = ./.;
      default-pkgs = p: py: with py; [ torch ];
      check-pkgs =
        p: py: with py; [
          hypothesis
          mypy
          pytest
        ];
      ci-pkgs =
        p: py: with py; [
          black
          coverage
        ];
      dev-pkgs =
        p: py: with py; [
          matplotlib
          python-lsp-server
        ];
      lookup-pkg-sets =
        ps: p: py:
        builtins.concatMap (f: f p py) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        pkgs.stdenv.mkDerivation {
          inherit pname version src;
          propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pkgs pypkgs;
          buildPhase = ":";
          installPhase = ''
            mkdir -p $out/${pypkgs.python.sitePackages}
            mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
          '';
        };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps pkgs)}/bin/python";
      in
      {
        apps.ci = {
          type = "app";
          program = "${
            let
              pname = "ci";
              python = python-with [
                default-pkgs
                check-pkgs
                ci-pkgs
              ];
              find = "${pkgs.findutils}/bin/find";
              nixfmt-bin = "${nixfmt.packages.${system}.default}/bin/nixfmt";
              rm = "${pkgs.coreutils}/bin/rm";
              xargs = "${pkgs.findutils}/bin/xargs";
              exec = ''
                #!${pkgs.bash}/bin/bash

                set -eu

                ${rm} -fr result
                ${find} . -name '*.nix' | ${xargs} ${nixfmt-bin} --check
                ${python} -m black --check .
                ${python} -m mypy .

                ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
                ${python} -m coverage report -m --fail-under=100
              '';
            in
            pkgs.stdenv.mkDerivation {
              inherit pname version src;
              buildPhase = ":";
              installPhase = ''
                mkdir -p $out/bin
                echo "${exec}" > $out/bin/${pname}
                chmod +x $out/bin/${pname}
              '';
            }
          }/bin/ci";
        };
        devShells.default = pkgs.mkShell {
          packages = (
            lookup-pkg-sets [
              default-pkgs
              check-pkgs
              ci-pkgs
              dev-pkgs
            ] pkgs pypkgs
          );
        };
      }
    );
}
