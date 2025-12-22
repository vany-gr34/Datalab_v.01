from io import BytesIO
import base64

def figs_to_html(figs):
    """
    Converts matplotlib figures to Dash HTML <img> elements
    """
    img_elements = []
    for name, fig in figs.items():
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        img_elements.append({'name': name, 'img': f'data:image/png;base64,{encoded}'})
    return img_elements
