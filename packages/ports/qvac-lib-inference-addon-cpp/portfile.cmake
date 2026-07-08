# QVAC-21914 — VALIDATION-ONLY overlay port, NOT FOR MERGE.
#
# Builds qvac-lib-inference-addon-cpp from the in-tree monorepo source
# (packages/inference-addon-cpp) instead of the registry-pinned 1.2.3, so
# llm-llamacpp's prebuild is compiled against the JsAsyncTask cancel/unload
# teardown fix without a registry publish. This mirrors the registry portfile's
# cmake flow; the only difference is the source comes from a local copy rather
# than vcpkg_from_git. The permanent fix ships via a qvac-registry-vcpkg version
# bump + consumer pin bump (see rollout-fabric-change / rollout-phase-* skills),
# after which this overlay and the "overlay-ports" entry are removed.

set(SRC_IN "${CMAKE_CURRENT_LIST_DIR}/../../inference-addon-cpp")

# The package CMakeLists writes .clang-format/.clang-tidy/.git hook files into
# its own source dir at configure time, so copy into a throwaway buildtree dir
# and never point cmake at the live monorepo checkout.
set(SOURCE_PATH "${CURRENT_BUILDTREES_DIR}/${PORT}-local")
file(REMOVE_RECURSE "${SOURCE_PATH}")
file(COPY "${SRC_IN}/" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    -DBUILD_TESTING=OFF
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

file(
  INSTALL "${SRC_IN}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright
)
