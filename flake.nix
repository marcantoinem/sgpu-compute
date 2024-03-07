{
  description = "wgpu template";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });

    in
    {
      devShells = forEachSupportedSystem
        ({ pkgs }: {
          default =
            pkgs.mkShell
              ({
                AMD_VULKAN_ICD = "RADV";
                LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${with pkgs; lib.makeLibraryPath [
                  udev alsa-lib vulkan-loader libxkbcommon wayland # To use wayland feature
                ]}";
                packages =
                  with pkgs; [
                    lld
                    vulkan-loader
                    udev
                    wayland
                    libxkbcommon
                    gnumake
                    jq
                  ];
              });
        });
    };
}
