#!/usr/bin/env bash
# Install the tested Debian 13 amd64 QE/SSSP packages without root access.
# No system APT configuration, packages, or repository files are modified.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [installation-prefix; default /workspace/tools]" >&2
    exit 2
fi
prefix=$(realpath -m -- "${1:-/workspace/tools}")
case "$prefix" in
    *'"'*|*'\'*|*$'\n'*) echo "Prefix cannot contain quotes, backslashes or newlines." >&2; exit 2 ;;
esac
. /etc/os-release
if [[ ${ID:-} != debian || ${VERSION_ID:-} != 13 || $(dpkg --print-architecture) != amd64 ]]; then
    echo "This installer supports Debian 13 amd64 only. Install QE separately on other systems." >&2
    exit 2
fi
for tool in /usr/bin/apt-get /usr/bin/apt-cache dpkg-deb sha256sum ldd; do
    command -v "$tool" >/dev/null
done
keyring=/usr/share/keyrings/debian-archive-keyring.gpg
[[ -r "$keyring" ]] || { echo "Debian archive keyring is required." >&2; exit 2; }

apt_root="$prefix/apt"
qe_root="$prefix/quantum-espresso"
mkdir -p "$apt_root"/{etc/apt.conf.d,lists/partial,cache/archives/partial,packages} "$qe_root" "$prefix/bin"
touch "$apt_root/etc/main.conf"
cat > "$apt_root/etc/sources.list" <<EOF
deb [signed-by=$keyring] https://deb.debian.org/debian trixie main
EOF
# APT_CONFIG is read before global fragments, so host post-invoke hooks cannot
# write to system directories. Signature, TLS and package hash checks stay on.
cat > "$apt_root/apt.conf" <<EOF
Dir::Etc::parts "$apt_root/etc/apt.conf.d";
Dir::Etc::main "$apt_root/etc/main.conf";
Dir::Etc::sourcelist "$apt_root/etc/sources.list";
Dir::Etc::sourceparts "-";
Dir::State::lists "$apt_root/lists";
Dir::Cache "$apt_root/cache";
Dir::Cache::archives "$apt_root/cache/archives";
APT::Get::List-Cleanup "0";
APT::Sandbox::User "$(id -un)";
EOF
export APT_CONFIG="$apt_root/apt.conf"
/usr/bin/apt-get update
(
    cd "$apt_root/packages"
    /usr/bin/apt-get download \
        quantum-espresso=6.7-3+b1 \
        libscalapack-openmpi2.2=2.2.2-1 \
        quantum-espresso-data-sssp=1.3.0-3
    # Pin artifacts as well as versions. These hashes were checked against the
    # signed Debian trixie package index. Never replace them to bypass a failure.
    sha256sum --check <<'EOF'
44f9f38afc08fa53148cdddf8936464ba992b894a0843cc4aaf8749809b54bca  quantum-espresso_6.7-3+b1_amd64.deb
19edff3e0d4d27b350965246ac93de8bc0328fa3c42f1b9a93048873ec2aeb7d  libscalapack-openmpi2.2_2.2.2-1_amd64.deb
f479bb38fa365d1a189610f0c1cdc116771d4228e7322e943c3352ff61c8ebbf  quantum-espresso-data-sssp_1.3.0-3_all.deb
EOF
    for package in quantum-espresso_6.7-3+b1_amd64.deb \
        libscalapack-openmpi2.2_2.2.2-1_amd64.deb \
        quantum-espresso-data-sssp_1.3.0-3_all.deb; do
        dpkg-deb --extract "$package" "$qe_root"
    done
)
/usr/bin/apt-cache show quantum-espresso=6.7-3+b1 \
    libscalapack-openmpi2.2=2.2.2-1 quantum-espresso-data-sssp=1.3.0-3 \
    > "$apt_root/package-metadata.txt"

cat > "$prefix/bin/pw.x" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
prefix=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export LD_LIBRARY_PATH="$prefix/quantum-espresso/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$prefix/quantum-espresso/usr/bin/pw.x" "$@"
EOF
chmod +x "$prefix/bin/pw.x"
cat > "$prefix/bin/pw-mpi.x" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
prefix=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tasks=${ATLAS_MPI_TASKS:-4}
if [[ ! $tasks =~ ^[1-9][0-9]*$ ]]; then
    echo "ATLAS_MPI_TASKS must be a positive integer." >&2
    exit 2
fi
if ! mpi_launcher=$(command -v mpirun); then
    echo "The MPI wrapper requires mpirun on PATH." >&2
    exit 127
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec "$mpi_launcher" --bind-to none -np "$tasks" "$prefix/bin/pw.x" "$@"
EOF
chmod +x "$prefix/bin/pw-mpi.x"
LD_LIBRARY_PATH="$qe_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ldd "$qe_root/usr/bin/pw.x" > "$prefix/qe-libraries.txt"
if [[ $(< "$prefix/qe-libraries.txt") == *"not found"* ]]; then
    cat "$prefix/qe-libraries.txt" >&2
    echo "Install the missing Debian runtime libraries before running QE." >&2
    exit 1
fi
echo "QE executable: $prefix/bin/pw.x"
echo "MPI executable: $prefix/bin/pw-mpi.x (ATLAS_MPI_TASKS defaults to 4; OMP_NUM_THREADS defaults to 1)"
echo "SSSP 1.3.0 UPFs: $qe_root/usr/share/espresso/pseudo"
echo "Run scripts/validate_qe.py for a real calculation; library resolution alone is not validation."
