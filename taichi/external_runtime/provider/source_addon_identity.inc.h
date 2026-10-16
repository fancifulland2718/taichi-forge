// Shared cold identity check for separately built source addons. Included
// after Runtime and the platform loader; no device or compiler dependency.
bool identify_loaded_source_addon(Runtime &runtime) {
  try {
    std::filesystem::path path;
#if defined(_WIN32)
    std::vector<wchar_t> buffer(256);
    for (;;) {
      const auto count = GetModuleFileNameW(runtime.library, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
      if (!count)
        throw std::runtime_error("cannot identify loaded addon module");
      if (count < buffer.size()) {
        path = std::wstring(buffer.data(), count);
        break;
      }
      if (buffer.size() > std::numeric_limits<DWORD>::max() / 2)
        throw std::runtime_error("loaded addon path exceeds platform ABI");
      buffer.resize(buffer.size() * 2);
    }
#else
    Dl_info info{};
    if (!dladdr(load_symbol(runtime.library, kRequiredSymbols[0]), &info) ||
        !info.dli_fname)
      throw std::runtime_error("cannot identify loaded addon module");
    path = info.dli_fname;
#endif
    path = std::filesystem::canonical(path);
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot read loaded addon identity");
    picosha2::hash256_one_by_one hasher;
    std::array<char, 16384> buffer_bytes;
    while (input) {
      input.read(buffer_bytes.data(), buffer_bytes.size());
      hasher.process(buffer_bytes.begin(),
                     buffer_bytes.begin() + input.gcount());
    }
    if (!input.eof())
      throw std::runtime_error("loaded addon identity read failed");
    hasher.finish();
    runtime.library_path = path.u8string();
    runtime.build_version = "sha256:" + picosha2::get_hash_hex_string(hasher);
    return true;
  } catch (const std::exception &error) {
    last_error = error.what();
    return false;
  }
}
