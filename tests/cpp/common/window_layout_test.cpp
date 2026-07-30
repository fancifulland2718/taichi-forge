#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "taichi/ui/common/window_layout.h"

namespace taichi::ui {

namespace {

WindowEdgeRegionConfig region(float size,
                              float minimum_size,
                              float maximum_fraction = 1.0f) {
  WindowEdgeRegionConfig result;
  result.enabled = true;
  result.size = size;
  result.minimum_size = minimum_size;
  result.maximum_fraction = maximum_fraction;
  return result;
}

}  // namespace

TEST(WindowLayoutTest, FullWindowIsDefaultRenderViewport) {
  WindowLayoutState layout;
  layout.update_dimensions(800.0f, 600.0f, 1600, 1200);

  const auto &snapshot = layout.snapshot();
  EXPECT_FLOAT_EQ(snapshot.render_viewport.x0, 0.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.y0, 0.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.x1, 800.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.y1, 600.0f);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.x1, 1600);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.y1, 1200);
}

TEST(WindowLayoutTest, FourEdgesProduceOneCentralViewport) {
  WindowLayoutState layout;
  layout.set_minimum_render_size(320.0f, 240.0f);
  layout.configure_region(WindowEdge::top, region(60.0f, 30.0f));
  layout.configure_region(WindowEdge::bottom, region(40.0f, 20.0f));
  layout.configure_region(WindowEdge::left, region(180.0f, 120.0f));
  layout.configure_region(WindowEdge::right, region(220.0f, 140.0f));
  layout.update_dimensions(1000.0f, 700.0f, 2000, 1400);

  const auto &snapshot = layout.snapshot();
  EXPECT_FLOAT_EQ(snapshot.render_viewport.x0, 180.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.y0, 60.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.x1, 780.0f);
  EXPECT_FLOAT_EQ(snapshot.render_viewport.y1, 660.0f);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.x0, 360);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.y0, 120);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.x1, 1560);
  EXPECT_EQ(snapshot.framebuffer_render_viewport.y1, 1320);
}

TEST(WindowLayoutTest, OppositeEdgesYieldToMinimumRenderSize) {
  WindowLayoutState layout;
  layout.set_minimum_render_size(400.0f, 300.0f);
  layout.configure_region(WindowEdge::top, region(240.0f, 180.0f));
  layout.configure_region(WindowEdge::bottom, region(240.0f, 180.0f));
  layout.configure_region(WindowEdge::left, region(400.0f, 300.0f));
  layout.configure_region(WindowEdge::right, region(400.0f, 300.0f));
  layout.update_dimensions(800.0f, 600.0f, 800, 600);

  const auto &viewport = layout.snapshot().render_viewport;
  EXPECT_FLOAT_EQ(viewport.width(), 400.0f);
  EXPECT_FLOAT_EQ(viewport.height(), 300.0f);
  EXPECT_FLOAT_EQ(layout.snapshot()
                      .edge_regions[window_edge_index(WindowEdge::left)]
                      .width(),
                  200.0f);
  EXPECT_FLOAT_EQ(layout.snapshot()
                      .edge_regions[window_edge_index(WindowEdge::top)]
                      .height(),
                  150.0f);
}

TEST(WindowLayoutTest, CollapseAndRestorePreservePreferredSize) {
  WindowLayoutState layout;
  layout.configure_region(WindowEdge::right, region(260.0f, 120.0f));
  layout.update_dimensions(800.0f, 600.0f, 800, 600);
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x1, 540.0f);

  layout.set_region_collapsed(WindowEdge::right, true);
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x1, 800.0f);

  layout.set_region_collapsed(WindowEdge::right, false);
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x1, 540.0f);
}

TEST(WindowLayoutTest, InteractiveUpdatesCommitAtFrameBoundary) {
  WindowLayoutState layout;
  layout.configure_region(WindowEdge::left, region(200.0f, 100.0f));
  layout.update_dimensions(800.0f, 600.0f, 800, 600);

  layout.request_region_resize(WindowEdge::left, 40.0f);
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x0, 200.0f);
  layout.apply_pending_updates();
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x0, 240.0f);

  layout.request_region_collapsed(WindowEdge::left, true);
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x0, 240.0f);
  layout.apply_pending_updates();
  EXPECT_FLOAT_EQ(layout.snapshot().render_viewport.x0, 0.0f);
}

TEST(WindowLayoutTest, CursorMappingUsesCentralViewport) {
  WindowLayoutState layout;
  layout.configure_region(WindowEdge::top, region(100.0f, 20.0f));
  layout.configure_region(WindowEdge::left, region(200.0f, 20.0f));
  layout.update_dimensions(1000.0f, 800.0f, 1000, 800);

  EXPECT_FALSE(layout.cursor_in_render_viewport(0.1f, 0.5f));
  EXPECT_FALSE(layout.cursor_in_render_viewport(0.5f, 0.95f));
  EXPECT_TRUE(layout.cursor_in_render_viewport(0.6f, 0.5f));

  const auto center =
      layout.render_cursor_position(0.6f, 0.4375f, false);
  EXPECT_FLOAT_EQ(center.first, 0.5f);
  EXPECT_FLOAT_EQ(center.second, 0.5f);
  const auto clamped =
      layout.render_cursor_position(0.0f, 1.0f, true);
  EXPECT_FLOAT_EQ(clamped.first, 0.0f);
  EXPECT_FLOAT_EQ(clamped.second, 1.0f);
}

TEST(WindowLayoutTest, InvalidConfigurationFailsClosed) {
  WindowLayoutState layout;
  auto invalid = region(100.0f, 20.0f);
  invalid.maximum_fraction = 1.1f;
  EXPECT_THROW(layout.configure_region(WindowEdge::left, invalid),
               std::invalid_argument);

  invalid = region(100.0f, 20.0f);
  invalid.size = std::nanf("");
  EXPECT_THROW(layout.configure_region(WindowEdge::left, invalid),
               std::invalid_argument);
  EXPECT_THROW(layout.update_dimensions(10.0f, 10.0f, -1, 10),
               std::invalid_argument);
  EXPECT_THROW(parse_window_edge("center"), std::invalid_argument);
}

TEST(WindowLayoutTest, RepeatedResizeCollapseAndDpiChangesStayBounded) {
  WindowLayoutState layout;
  layout.set_minimum_render_size(160.0f, 120.0f);
  for (auto edge :
       {WindowEdge::top, WindowEdge::bottom, WindowEdge::left,
        WindowEdge::right}) {
    layout.configure_region(edge, region(180.0f, 40.0f, 0.6f));
  }

  for (int iteration = 0; iteration < 1000; ++iteration) {
    const auto edge = static_cast<WindowEdge>(iteration % kWindowEdgeCount);
    layout.request_region_resize(edge,
                                 iteration % 2 == 0 ? 3.0f : -2.0f);
    if (iteration % 17 == 0) {
      layout.request_region_collapsed(edge, true);
    } else if (iteration % 17 == 1) {
      layout.request_region_collapsed(edge, false);
    }
    layout.apply_pending_updates();
    const float logical_width = 320.0f + float(iteration % 701);
    const float logical_height = 240.0f + float(iteration % 461);
    layout.update_dimensions(
        logical_width, logical_height,
        static_cast<int>(std::lround(logical_width * 1.5f)),
        static_cast<int>(std::lround(logical_height * 1.5f)));

    const auto &snapshot = layout.snapshot();
    EXPECT_GE(snapshot.render_viewport.x0, 0.0f);
    EXPECT_GE(snapshot.render_viewport.y0, 0.0f);
    EXPECT_LE(snapshot.render_viewport.x1, logical_width);
    EXPECT_LE(snapshot.render_viewport.y1, logical_height);
    EXPECT_GE(snapshot.framebuffer_render_viewport.width(), 0);
    EXPECT_GE(snapshot.framebuffer_render_viewport.height(), 0);
  }
}

}  // namespace taichi::ui
