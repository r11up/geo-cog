"""Hand-drawn SVG icons for the diagrams (no third-party artwork)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets"
CLASS = {"neutral": "#9E9E9E", "sadness": "#4A7FB5", "anger": "#D6604D", "happiness": "#F2C14E",
         "calmness": "#6A9F4B"}
INK = "#1F2328"


def svg(body, w=100, h=100):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'


def face(kind):
    c = CLASS[kind]
    eyes = '<circle cx="36" cy="42" r="5" fill="{0}"/><circle cx="64" cy="42" r="5" fill="{0}"/>'.format(INK)
    mouth = {"neutral": '<line x1="34" y1="68" x2="66" y2="68" stroke="{0}" stroke-width="5" stroke-linecap="round"/>',
             "happiness": '<path d="M30 60 Q50 82 70 60" fill="none" stroke="{0}" stroke-width="5" stroke-linecap="round"/>',
             "sadness": '<path d="M32 74 Q50 56 68 74" fill="none" stroke="{0}" stroke-width="5" stroke-linecap="round"/>',
             "anger": '<path d="M32 74 Q50 60 68 74" fill="none" stroke="{0}" stroke-width="5" stroke-linecap="round"/>',
             "calmness": '<path d="M36 66 Q50 74 64 66" fill="none" stroke="{0}" stroke-width="5" stroke-linecap="round"/>'}[kind]
    extra = ""
    if kind == "anger":
        extra = ('<line x1="26" y1="30" x2="44" y2="37" stroke="{0}" stroke-width="5" stroke-linecap="round"/>'
                 '<line x1="74" y1="30" x2="56" y2="37" stroke="{0}" stroke-width="5" stroke-linecap="round"/>')
    if kind == "sadness":
        extra = '<path d="M68 50 q4 8 0 12 q-4 -4 0 -12z" fill="#8DB6E0"/>'
    if kind == "calmness":
        eyes = ('<path d="M29 43 Q36 49 43 43" fill="none" stroke="{0}" stroke-width="4" stroke-linecap="round"/>'
                '<path d="M57 43 Q64 49 71 43" fill="none" stroke="{0}" stroke-width="4" stroke-linecap="round"/>')
    body = f'<circle cx="50" cy="50" r="44" fill="{c}" stroke="{INK}" stroke-width="3"/>' + eyes + mouth + extra
    return svg(body.format(INK))


def satellite():
    body = ('<g transform="rotate(-30 50 50)">'
            '<rect x="40" y="38" width="20" height="24" rx="3" fill="#C9CED6" stroke="#1F2328" stroke-width="2.5"/>'
            '<rect x="6" y="42" width="30" height="16" fill="#3A6EA5" stroke="#1F2328" stroke-width="2.5"/>'
            '<rect x="64" y="42" width="30" height="16" fill="#3A6EA5" stroke="#1F2328" stroke-width="2.5"/>'
            '<line x1="16" y1="42" x2="16" y2="58" stroke="#9CC3E6" stroke-width="1.5"/>'
            '<line x1="26" y1="42" x2="26" y2="58" stroke="#9CC3E6" stroke-width="1.5"/>'
            '<line x1="74" y1="42" x2="74" y2="58" stroke="#9CC3E6" stroke-width="1.5"/>'
            '<line x1="84" y1="42" x2="84" y2="58" stroke="#9CC3E6" stroke-width="1.5"/>'
            '<line x1="36" y1="50" x2="40" y2="50" stroke="#1F2328" stroke-width="2.5"/>'
            '<line x1="60" y1="50" x2="64" y2="50" stroke="#1F2328" stroke-width="2.5"/>'
            '<path d="M44 62 L50 72 L56 62" fill="#E9ECEF" stroke="#1F2328" stroke-width="2"/></g>'
            '<path d="M58 74 Q66 82 74 74" fill="none" stroke="#9E9E9E" stroke-width="2"/>'
            '<path d="M54 80 Q66 92 78 80" fill="none" stroke="#C9CED6" stroke-width="2"/>')
    return svg(body)


def viewer():
    """Person wearing an EEG cap looking at a monitor showing a street scene (composition stripes)."""
    body = ('<rect x="52" y="14" width="44" height="34" rx="2" fill="#FFFFFF" stroke="#1F2328" stroke-width="2.5"/>'
            '<rect x="55" y="17" width="38" height="9" fill="#BFD7EA"/>'
            '<rect x="55" y="26" width="12" height="14" fill="#8C8C8C"/><rect x="67" y="22" width="10" height="18" fill="#A6A6A6"/>'
            '<rect x="77" y="24" width="16" height="16" fill="#6A9F4B"/>'
            '<rect x="55" y="40" width="38" height="5" fill="#5B5B5B"/>'
            '<rect x="70" y="48" width="8" height="8" fill="#1F2328"/><rect x="62" y="56" width="24" height="3" fill="#1F2328"/>'
            '<circle cx="24" cy="38" r="14" fill="#F1D3B6" stroke="#1F2328" stroke-width="2.5"/>'
            '<path d="M10 36 A14 14 0 0 1 38 36" fill="#4A7FB5" stroke="#1F2328" stroke-width="2.5"/>'
            '<circle cx="15" cy="30" r="1.8" fill="#FFFFFF"/><circle cx="24" cy="25" r="1.8" fill="#FFFFFF"/>'
            '<circle cx="33" cy="30" r="1.8" fill="#FFFFFF"/>'
            '<path d="M4 92 Q6 60 24 58 Q42 60 44 92 Z" fill="#7F8C9A" stroke="#1F2328" stroke-width="2.5"/>'
            '<path d="M38 40 L52 34" stroke="#9E9E9E" stroke-width="1.5" stroke-dasharray="3 2"/>')
    return svg(body)


def lab_person():
    """Participant in a lab conversation (EAV): EEG cap and speech bubble."""
    body = ('<circle cx="34" cy="40" r="16" fill="#F1D3B6" stroke="#1F2328" stroke-width="2.5"/>'
            '<path d="M18 38 A16 16 0 0 1 50 38" fill="#4A7FB5" stroke="#1F2328" stroke-width="2.5"/>'
            '<circle cx="26" cy="30" r="2" fill="#FFF"/><circle cx="34" cy="25" r="2" fill="#FFF"/><circle cx="42" cy="30" r="2" fill="#FFF"/>'
            '<path d="M12 96 Q14 62 34 60 Q54 62 56 96 Z" fill="#7F8C9A" stroke="#1F2328" stroke-width="2.5"/>'
            '<path d="M58 14 h34 a6 6 0 0 1 6 6 v14 a6 6 0 0 1 -6 6 h-22 l-8 8 v-8 h-4 a6 6 0 0 1 -6 -6 v-14 a6 6 0 0 1 6 -6z" '
            'fill="#FFFFFF" stroke="#1F2328" stroke-width="2.5"/>'
            '<circle cx="66" cy="27" r="2.5" fill="#1F2328"/><circle cx="75" cy="27" r="2.5" fill="#1F2328"/>'
            '<circle cx="84" cy="27" r="2.5" fill="#1F2328"/>')
    return svg(body)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for k in CLASS:
        (OUT / f"face_{k}.svg").write_text(face(k))
    (OUT / "satellite.svg").write_text(satellite())
    (OUT / "viewer.svg").write_text(viewer())
    (OUT / "lab_person.svg").write_text(lab_person())
    print(sorted(p.name for p in OUT.glob("*.svg")))
