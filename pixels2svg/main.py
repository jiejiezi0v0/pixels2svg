from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import base64
import cc3d
import numpy as np
from svgwrite.container import Group
import PIL
from pixels2svg.utils import geometry, pixel, preprocessing, svg
from pixels2svg.utils.geometry import Contours

T = TypeVar('T')


def find_contours(
        rgba_array: np.ndarray,
        group_by_color: bool = False
) -> Union[Dict[pixel.PixelRGBA, Tuple[Contours, ...]],
           Tuple[Tuple[Contours, pixel.PixelRGBA], ...]]:
    id_array = pixel.rgba_array_to_id_array(rgba_array)
    labels = cc3d.connected_components(id_array,
                                       out_dtype=np.uint64,
                                       connectivity=4)
    if group_by_color:
        all_contours: dict = {}
        result = OrderedDict()
    else:
        all_contours: list = []
        result = tuple()

    def sort_contours_by_area(
            contours_list: List[Tuple[T, int]]) -> Tuple[T, ...]:
        sorted_contours = sorted(contours_list,
                                 key=lambda c: c[1],
                                 reverse=True)
        return tuple(c[0] for c in sorted_contours)

    for blob_id, blob_shape in cc3d.each(labels, binary=True, in_place=True):

        pixel_coord = geometry.find_first_non_zero_coords(blob_shape)
        color = tuple(rgba_array[pixel_coord])
        # ignore transparent pixels
        if color[3] == 0:
            continue

        contours: Contours = geometry.calculate_blob_contours(blob_shape)
        shape_area = blob_shape.sum()

        if group_by_color:
            all_contours: dict
            if color in all_contours:
                all_contours[color].append((contours, shape_area))
            else:
                all_contours[color] = [(contours, shape_area)]
        else:
            all_contours: list
            all_contours.append(((contours, color), shape_area))

        if group_by_color:
            sorted_colors = sort_contours_by_area(
                [(color, sum(c[1] for c in contours_list))
                 for color, contours_list in all_contours.items()]
            )
            result = OrderedDict()
            for color in sorted_colors:
                result[color] = sort_contours_by_area(all_contours[color])
        else:
            result = sort_contours_by_area(all_contours)

    return result


def trace_pixel_polygons_as_svg(rgba_array: np.ndarray, group_by_color: bool = False) -> svg.Drawing:
    traced_contours = find_contours(rgba_array, group_by_color)
    svg_img = svg.Drawing(rgba_array.shape[0], rgba_array.shape[1])
    
    # Track used colors to reuse existing groups
    color_groups = {}
    
    def get_minimal_color_id(color: pixel.PixelRGBA) -> str:
        # Use base64 encoding for ultra-compact color representation
        return str(base64.b64encode(bytes(color)), 'utf-8')[:6]
    
    if group_by_color:
        for color, contours_tuple in traced_contours.items():
            color_hex = pixel.rgb_color_to_hex_code(color[:3])
            opacity = color[3] / 255
            
            # Create ultra-compact path data
            paths = []
            for contour in contours_tuple:
                # Optimize coordinates by removing redundant precision
                path = f"M{','.join(f'{int(x)},{int(y)}' for x,y in contour.outside)}Z"
                if contour.inner_holes:
                    for hole in contour.inner_holes:
                        path += f"M{','.join(f'{int(x)},{int(y)}' for x,y in hole)}Z"
                paths.append(path)
            
            # Create or reuse group
            cid = get_minimal_color_id(color)
            if cid not in color_groups:
                g = Group()
                if opacity < 1:
                    g.update({'fill': color_hex, 'fill-opacity': f"{opacity:.2f}"})
                else:
                    g.update({'fill': color_hex})
                color_groups[cid] = g
                svg_img.add(g)
            
            # Add paths to group with minimal attributes
            for path in paths:
                color_groups[cid].add(Path(d=path))
    else:
        # Process individual contours with minimal attributes
        for contour, color in traced_contours:
            path = f"M{','.join(f'{int(x)},{int(y)}' for x,y in contour.outside)}Z"
            if contour.inner_holes:
                for hole in contour.inner_holes:
                    path += f"M{','.join(f'{int(x)},{int(y)}' for x,y in hole)}Z"
            
            attrs = {'d': path, 'fill': pixel.rgb_color_to_hex_code(color[:3])}
            if color[3] < 255:
                attrs['fill-opacity'] = f"{color[3]/255:.2f}"
            svg_img.add(Path(**attrs))

    return svg_img


def pixels2svg(input_path: Optional[str] = None,
               bitmap_content: Optional[PIL.Image.Image] = None,
               output_path: Optional[str] = None,
               group_by_color: bool = True,
               color_tolerance: int = 0,
               remove_background: bool = False,
               background_tolerance: float = 1.0,
               maximal_non_bg_artifact_size: float = 2.0,
               as_string: bool = False,
               pretty: bool = True,) -> Optional[Union[svg.Drawing, str]]:
    """
    Parameters
    ----------
    input_path: str
        Path of the input bitmap image
    output_path: Optional[str]
        Path of the output SVG image (optional).
        If passed, the function will return None.
        If not passed, the function will return the SVG data as a `str` or a
        `Drawing` depending on the `as_string` parameter.
    group_by_color: bool
        If True (default), group same-color shapes under <g> SVG elements.
    color_tolerance: int
        Optional tolerance parameter that defines if adjacent pixels of
        close colors should be merged together in a single SVG shape.
        Tolerance is applied based on luminosity.
        1 represents the smallest difference of luminosity, i.e. a difference
        of 1 in the Blue channel.
    remove_background: bool
        If True, tries to remove the background before the conversion to SVG
        (default False). Simple technique based on contour detection,
        probably won't work well with complex images.
    background_tolerance: float
        (Only relevant when `remove_background = True`)
        Arbitrary quantity of blur use to remove noise - just fine-tune the
        value if the default (1.0) doesn't work well. 0 means no blur.
    maximal_non_bg_artifact_size: float
        (Only relevant when `remove_background = True`)
        When a blob of pixels is clone enough to the detected image contours,
        and below this threshold, it won't be considered as background.
        Combined with `background_tolerance`, this allows you to control how
        progressive the background detection should be with blurred contours.
        Size is expressed in % of total image pixels.
    as_string: bool
        If True and no `output_path` is passed, return a `str` representing
        the SVG data. (default False)
    pretty: bool
        If True (default), output SVG code is pretty-printed.

    Returns
    -------
    Optional[Union[svg.Drawing, str]]
        Depends on the `output_path` and `as_string` parameters
    """
    if bitmap_content is not None:
        img_rgba_array = pixel.read_image(bitmap_content)
    else:
        img_rgba_array = pixel.read_image(input_path)

    if color_tolerance > 0:
        img_rgba_array = preprocessing.apply_color_tolerance(
            img_rgba_array,
            color_tolerance)

    if remove_background:
        img_rgba_array = preprocessing.remove_background(
            img_rgba_array,
            background_tolerance=background_tolerance,
            maximal_non_bg_artifact_size=maximal_non_bg_artifact_size)

    svg_drawing = trace_pixel_polygons_as_svg(img_rgba_array, group_by_color)

    if output_path:
        svg_drawing.save_to_path(output_path, pretty)
    else:
        if as_string:
            return svg_drawing.to_string(pretty)
        else:
            return svg_drawing
