package = "dstoptim"
version = "scm-1"

source = {
    url = "git://github.com/malt2/dstoptim"
}


description = {
   summary = "dstoptim for malt2 for torch",
   detailed = [[
   	    dstoptim - malt2 torch package
   ]],
   homepage = "https://github.com/malt2/dstoptim"
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
