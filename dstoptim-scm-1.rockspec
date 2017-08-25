package = "dstoptim"
version = "scm-1"

source = {
    url = "file:///net/mlfs01/export/users/asim/malt-2/torch/dstoptim"
}


description = {
   summary = "dstoptim for malt2 for torch",
   detailed = [[
   	    dstoptim - malt2 torch package
   ]],
   homepage = "https://mlsvn.nec-labs.com/projects/mlsvn/browser/milde_pkg/milde_malt2"
}

dependencies = {
   "torch >= 7.0",
   "malt-2",
   "optim"
}

build = {
    type = "command",
    build_command = [[
    cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
    ]],
    install_command = "cd build && $(MAKE) install"
}
