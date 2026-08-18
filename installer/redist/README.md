# VC++ Redistributable

Before compiling `installer/SMiCA.iss`, download the official Microsoft Visual C++
2015-2022 Redistributable (x64) and save it in this folder as `vc_redist.x64.exe`:

https://aka.ms/vs/17/release/vc_redist.x64.exe

This file isn't committed to the repository (it's a 25MB third-party binary, not part
of this project's source), so `installer/SMiCA.iss` will fail to compile until it's
present here.
