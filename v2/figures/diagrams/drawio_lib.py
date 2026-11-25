"""Tiny helper for writing editable draw.io diagrams from Python (layout kept in code)."""
from html import escape

FONT = "fontFamily=Helvetica;"
PALETTE = {
    "env": ("#EAF3E2", "#6A9F4B", "#2D4A1C"),       # fill, stroke, title text
    "neural": ("#E4EEF8", "#4A7FB5", "#1F3B57"),
    "link": ("#EFE8F6", "#8064A2", "#3D2A57"),
    "out": ("#FCEEE2", "#D9822B", "#5C2A08"),
    "grey": ("#F4F4F4", "#9E9E9E", "#333333"),
}


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.w, self.h, self.cells, self.n = name, width, height, [], 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def text(self, x, y, w, h, html, size=12, bold=False, align="center", color="#222222", valign="middle"):
        style = (f"text;html=1;whiteSpace=wrap;align={align};verticalAlign={valign};{FONT}"
                 f"fontSize={size};fontColor={color};" + ("fontStyle=1;" if bold else ""))
        return self._vertex(html, style, x, y, w, h)

    def box(self, x, y, w, h, html, kind="grey", size=12, fill=None, title=False, dashed=False,
            align="center", valign="middle", stroke_w=1.5, rounded=True):
        f, s, t = PALETTE[kind]
        style = (f"rounded={1 if rounded else 0};arcSize=8;whiteSpace=wrap;html=1;fillColor={fill or f};"
                 f"strokeColor={s};strokeWidth={stroke_w};{FONT}fontSize={size};fontColor={t if title else '#1A1A1A'};"
                 f"align={align};verticalAlign={valign};spacing=4;" + ("dashed=1;" if dashed else ""))
        if title:
            style += "fontStyle=1;spacingTop=6;"
        return self._vertex(html, style, x, y, w, h)

    def panel(self, x, y, w, h, title, kind, size=14):
        f, s, t = PALETTE[kind]
        style = (f"rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};strokeWidth=2;"
                 f"verticalAlign=top;align=center;{FONT}fontSize={size};fontStyle=1;fontColor={t};spacingTop=6;")
        return self._vertex(title, style, x, y, w, h)

    def arrow(self, src=None, dst=None, pts=None, color="#333333", width=2, dashed=False,
              exit=None, entry=None, label="", end="block", label_bg="#FFFFFF", via=None):
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow={end};endFill=1;"
                 f"labelBackgroundColor={label_bg};"
                 f"strokeColor={color};strokeWidth={width};{FONT}fontSize=11;" + ("dashed=1;" if dashed else ""))
        if exit:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        cid = self._id()
        attrs = f'id="{cid}" value="{escape(label)}" style="{style}" edge="1" parent="1"'
        if src:
            attrs += f' source="{src}"'
        if dst:
            attrs += f' target="{dst}"'
        geo = '<mxGeometry relative="1" as="geometry">'
        if pts:
            geo += f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/><mxPoint x="{pts[1][0]}" y="{pts[1][1]}" as="targetPoint"/>'
        if via:
            geo += '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in via) + "</Array>"
        geo += "</mxGeometry>"
        self.cells.append(f"<mxCell {attrs}>{geo}</mxCell>")
        return cid

    def _vertex(self, html, style, x, y, w, h):
        cid = self._id()
        self.cells.append(f'<mxCell id="{cid}" value="{escape(html)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def save(self, path):
        body = "\n".join(self.cells)
        xml = (f'<mxfile host="drawio"><diagram id="d" name="{escape(self.name)}">'
               f'<mxGraphModel grid="0" page="1" pageWidth="{self.w}" pageHeight="{self.h}" background="#FFFFFF">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>\n{body}\n</root></mxGraphModel></diagram></mxfile>')
        open(path, "w").write(xml)


def _image(self, x, y, w, h, path, border=None):
    """Embed a PNG or SVG file as an image cell (data URI, so the .drawio file is self-contained)."""
    import base64
    from pathlib import Path
    p = Path(path)
    mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    style = (f"shape=image;html=1;imageAspect=1;aspect=fixed;verticalLabelPosition=bottom;"
             f"image=data:{mime},{b64};" + (f"imageBorder={border};" if border else ""))
    return self._vertex("", style, x, y, w, h)


Diagram.image = _image
