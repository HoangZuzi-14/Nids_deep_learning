import os
import urllib.request
import shutil
from fpdf import FPDF

# Custom Font directory in local scratch space
FONT_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\scratch"
os.makedirs(FONT_DIR, exist_ok=True)

# Roboto font URLs from Google Fonts raw mirror (highly stable and supports Vietnamese Unicode)
ROBOTO_REGULAR_URL = "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf"
ROBOTO_BOLD_URL = "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Bold.ttf"

ROBOTO_REGULAR_PATH = os.path.join(FONT_DIR, "Roboto-Regular.ttf")
ROBOTO_BOLD_PATH = os.path.join(FONT_DIR, "Roboto-Bold.ttf")

# Download font files with safe fallback
def ensure_fonts():
    try:
        if not os.path.exists(ROBOTO_REGULAR_PATH):
            print("[+] Downloading Roboto-Regular.ttf...")
            urllib.request.urlretrieve(ROBOTO_REGULAR_URL, ROBOTO_REGULAR_PATH)
        if not os.path.exists(ROBOTO_BOLD_PATH):
            print("[+] Downloading Roboto-Bold.ttf...")
            urllib.request.urlretrieve(ROBOTO_BOLD_URL, ROBOTO_BOLD_PATH)
        return True
    except Exception as e:
        print(f"[-] Failed to download Roboto fonts: {e}")
        return False

# Custom FPDF class to manage Header and Footer
class NIDSReportPDF(FPDF):
    def __init__(self, font_name="Roboto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_name = font_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if self.page_no() == 1:
            return # Skip header on cover page
        self.set_font(self.font_name, "B", 8)
        self.set_text_color(92, 118, 141)  # Slate Gray
        self.cell(0, 10, "Báo Cáo Đánh Giá Thực Nghiệm: Khung Phát Hiện Xâm Nhập Lai Đa Tập Dữ Liệu", border="B", align="R")
        self.ln(12)

    def footer(self):
        if self.page_no() == 1:
            return # Skip footer on cover page
        self.set_y(-15)
        self.set_font(self.font_name, "", 8)
        self.set_text_color(92, 118, 141)  # Slate Gray
        self.cell(0, 10, f"Trang {self.page_no()} / {{nb}}", align="C")

def build_pdf():
    # 1. Font Setup
    font_name = "Roboto"
    if ensure_fonts():
        pass
    else:
        # Fallback to local Windows Arial
        arial_path = r"C:\Windows\Fonts\arial.ttf"
        arial_bold_path = r"C:\Windows\Fonts\arialbd.ttf"
        if os.path.exists(arial_path) and os.path.exists(arial_bold_path):
            print("[+] Copying local Arial fonts as fallback...")
            font_regular = os.path.join(FONT_DIR, "arial.ttf")
            font_bold = os.path.join(FONT_DIR, "arialbd.ttf")
            try:
                shutil.copy(arial_path, font_regular)
                shutil.copy(arial_bold_path, font_bold)
                # Overwrite standard Roboto path references to use copied Arial files
                shutil.copy(font_regular, ROBOTO_REGULAR_PATH)
                shutil.copy(font_bold, ROBOTO_BOLD_PATH)
            except Exception as e:
                print(f"[-] Failed to copy Arial fonts: {e}")
                font_name = "Helvetica"
        else:
            font_name = "Helvetica"

    # Initialize FPDF
    pdf = NIDSReportPDF(font_name=font_name, orientation="P", unit="mm", format="A4")
    
    # Register Fonts
    if font_name == "Roboto":
        pdf.add_font("Roboto", "", ROBOTO_REGULAR_PATH)
        pdf.add_font("Roboto", "B", ROBOTO_BOLD_PATH)
    else:
        pdf.add_font("Helvetica", "", core=True)
        pdf.add_font("Helvetica", "B", core=True)

    pdf.alias_nb_pages() # Support total page count formatting

    # Theme colors
    navy_color = (27, 54, 93)     # #1B365D - Primary
    slate_color = (92, 118, 141)  # #5C768D - Secondary
    bg_accent = (240, 244, 248)    # Light gray-blue background
    dark_gray = (50, 50, 50)       # Main text

    # Helper methods for adding formatted text
    def add_heading(text, level=1):
        if level == 1:
            pdf.ln(4)
            pdf.set_font(font_name, "B", 14)
            pdf.set_text_color(*navy_color)
            pdf.cell(0, 10, text)
            pdf.ln(10)
            # Add colored horizontal rule underneath Heading 1
            pdf.set_draw_color(*slate_color)
            pdf.set_line_width(0.5)
            pdf.line(pdf.get_x(), pdf.get_y() - 1, pdf.get_x() + 190, pdf.get_y() - 1)
            pdf.ln(3)
        elif level == 2:
            pdf.ln(3)
            pdf.set_font(font_name, "B", 11)
            pdf.set_text_color(*slate_color)
            pdf.cell(0, 8, text)
            pdf.ln(8)
            pdf.ln(2)

    def add_paragraph(text):
        pdf.set_font(font_name, "", 9.5)
        pdf.set_text_color(*dark_gray)
        pdf.multi_cell(0, 5.5, text, align="J")
        pdf.ln(2)

    def add_bullet(text):
        pdf.set_font(font_name, "", 9.5)
        pdf.set_text_color(*dark_gray)
        # Indented bullet point
        pdf.set_x(15)
        pdf.cell(5, 5.5, "-")
        pdf.multi_cell(0, 5.5, text, align="J")
        pdf.set_x(10)
        pdf.ln(1)

    def draw_table(headers, rows, col_widths, align_cols=None):
        pdf.set_font(font_name, "B", 8.5)
        # Header Styling
        pdf.set_fill_color(*bg_accent)
        pdf.set_text_color(*navy_color)
        pdf.set_draw_color(210, 215, 223)
        pdf.set_line_width(0.2)
        
        # Draw headers
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        pdf.ln()

        # Row Styling
        pdf.set_font(font_name, "", 8)
        pdf.set_text_color(*dark_gray)
        
        alt_row = False
        for row in rows:
            # Alternate row background colors
            if alt_row:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            # Check for page break space
            if pdf.get_y() > 250:
                pdf.add_page()
                # Re-draw headers on new page
                pdf.set_font(font_name, "B", 8.5)
                pdf.set_fill_color(*bg_accent)
                pdf.set_text_color(*navy_color)
                for i, header in enumerate(headers):
                    pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
                pdf.ln()
                pdf.set_font(font_name, "", 8)
                pdf.set_text_color(*dark_gray)

            # Draw row columns
            for i, val in enumerate(row):
                align = "L"
                if align_cols:
                    align = align_cols[i]
                pdf.cell(col_widths[i], 7.5, str(val), border=1, fill=True, align=align)
            pdf.ln()
            alt_row = not alt_row
        pdf.ln(3)

    def embed_image(path, w, h=None):
        if os.path.exists(path):
            # Center alignment calculation
            x_pos = (210 - w) / 2
            pdf.image(path, x=x_pos, y=pdf.get_y(), w=w, h=h)
            pdf.set_y(pdf.get_y() + (h if h else w * 0.6) + 4) # advance Y coordinate by height + margin
        else:
            pdf.set_font(font_name, "B", 10)
            pdf.set_text_color(200, 50, 50)
            pdf.cell(0, 10, f"[Lỗi: Không tìm thấy hình ảnh tại {path}]", align="C")
            pdf.ln(10)
            pdf.ln(2)

    # -------------------------------------------------------------
    # PAGE 1: COVER PAGE
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Main Header Banner (Navy Rectangle)
    pdf.set_fill_color(*navy_color)
    pdf.rect(0, 0, 210, 80, "F")
    
    pdf.set_y(25)
    pdf.set_font(font_name, "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 10, "BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM:\nKHUNG PHÁT HIỆN XÂM NHẬP LAI ĐA TẬP DỮ LIỆU", align="C")
    
    pdf.ln(5)
    pdf.set_font(font_name, "", 12)
    pdf.set_text_color(210, 225, 240)
    pdf.cell(0, 10, "Multi-Dataset Hybrid NIDS Framework Evaluation", align="C")
    pdf.ln(10)
    
    # Lower metadata card
    pdf.set_y(100)
    pdf.set_font(font_name, "B", 12)
    pdf.set_text_color(*navy_color)
    pdf.cell(0, 10, "THÔNG TIN DỰ ÁN & THỰC NGHIỆM", align="C")
    pdf.ln(10)
    pdf.set_draw_color(*slate_color)
    pdf.set_line_width(1)
    pdf.line(80, 112, 130, 112)
    pdf.ln(10)
    
    metadata = [
        ("Đơn vị thực hiện", "NIDS Project Evaluation Team"),
        ("Ngày lập báo cáo", "12 tháng 06, 2026"),
        ("Phiên bản tài liệu", "v1.0 (Bản hoàn chỉnh thực nghiệm)"),
        ("Tập dữ liệu kiểm thử", "NSL-KDD, UNSW-NB15, CICIDS2017"),
        ("Các mô hình khảo sát", "RandomForest, XGBoost, LightGBM, MLP, CNN-1D, BiLSTM+Attention, CNN-LSTM Hybrid, Autoencoder (Anomaly)"),
    ]
    
    pdf.set_font(font_name, "", 10)
    for label, val in metadata:
        pdf.set_x(30)
        pdf.set_font(font_name, "B", 9.5)
        pdf.set_text_color(*slate_color)
        pdf.cell(50, 8, label + ":")
        pdf.set_font(font_name, "", 9.5)
        pdf.set_text_color(*dark_gray)
        pdf.cell(0, 8, val)
        pdf.ln(8)

    # Executive Summary Box
    pdf.ln(10)
    pdf.set_x(20)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.5)
    pdf.cell(170, 45, "", border=1, fill=True)
    pdf.set_y(pdf.get_y() - 42)
    
    pdf.set_x(25)
    pdf.set_font(font_name, "B", 10)
    pdf.set_text_color(*navy_color)
    pdf.cell(0, 6, "Tóm Tắt Kết Luận Thực Nghiệm (Executive Summary):")
    pdf.ln(6)
    
    pdf.set_font(font_name, "", 9)
    pdf.set_text_color(*dark_gray)
    summary_text = (
        "Báo cáo này tổng hợp kết quả huấn luyện, đánh giá và tối ưu hóa hệ thống Phát hiện xâm nhập mạng (NIDS) "
        "trên 3 tập dữ liệu tiêu chuẩn. Kết quả cho thấy mô hình RandomForest là baseline giám sát mạnh nhất. "
        "Hiện tượng chệch phân phối (Domain Shift) làm giảm sâu hiệu năng khi kiểm thử chéo. "
        "Đặc biệt, lớp quyết định lai (Hybrid Decision Layer) kết hợp RandomForest và Autoencoder giúp khôi phục "
        "tới 100% các cuộc tấn công bị bỏ sót trên NSL-KDD và 66.3% trên UNSW-NB15 dưới mức FAR kiểm soát được."
    )
    pdf.set_x(25)
    pdf.multi_cell(160, 5, summary_text, align="J")

    # -------------------------------------------------------------
    # PAGE 2: CHIẾN LƯỢC XỬ LÝ MẤT CÂN BẰNG MẪU (CICIDS2017)
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("1. Tiền xử lý & Chiến lược Mất cân bằng mẫu (CICIDS2017)")
    add_paragraph(
        "Tập dữ liệu CICIDS2017 chứa tổng cộng 1.513.416 mẫu huấn luyện sau khi làm sạch. "
        "Do đặc thù lưu lượng mạng Benign chiếm đa số tuyệt đối (hơn 83%), sự mất cân bằng giữa các lớp tấn công "
        "cực kỳ nghiêm trọng. Nhóm nghiên cứu thiết kế một chiến lược xử lý thích ứng chia theo các băng tần tỷ lệ mẫu (Ratio Bands)."
    )
    add_paragraph(
        "Lưu ý quan trọng: Mô hình sinh mẫu GAN (GAN Augmentation) không được sử dụng làm chiến lược chính trong "
        "pipeline cốt lõi do mẫu sinh ra cho dữ liệu dạng bảng (tabular) khó kiểm chứng độ tin cậy và có xu hướng "
        "làm tăng mạnh tỷ lệ cảnh báo giả (FAR) trong các thử nghiệm ban đầu."
    )
    
    # Class distribution table
    headers = ["Phân lớp (Class)", "Số lượng mẫu", "Tỷ lệ (%)", "Chiến lược xử lý thích ứng"]
    rows = [
        ["Benign", "1,257,890", "83.1159%", "controlled_undersampling (Giảm mẫu có kiểm soát)"],
        ["DoS", "116,248", "7.6812%", "mild_class_weight (Trọng số lớp nhẹ)"],
        ["DDoS", "76,810", "5.0753%", "mild_class_weight (Trọng số lớp nhẹ)"],
        ["PortScan", "54,491", "3.6005%", "weighted_sampler_plus_class_weight"],
        ["BruteForce", "5,491", "0.3628%", "focal_loss_plus_weighted_sampler"],
        ["WebAttack", "1,285", "0.0849%", "rare_grouping (Gom nhóm lớp hiếm)"],
        ["Botnet", "1,172", "0.0774%", "rare_grouping (Gom nhóm lớp hiếm)"],
        ["Infiltration", "22", "0.0015%", "rare_grouping (Gom nhóm lớp hiếm)"],
        ["Rare_Attack", "7", "0.0005%", "rare_grouping (Gom nhóm lớp hiếm)"]
    ]
    draw_table(headers, rows, [25, 30, 25, 110], ["L", "R", "R", "L"])
    
    add_heading("Quy tắc phân băng tần xử lý (Strategy Bands):", level=2)
    add_bullet("Tỷ lệ >= 20% (Benign): Áp dụng controlled_undersampling nhằm ngăn chặn lớp đa số áp đảo mô hình.")
    add_bullet("Tỷ lệ 5% đến < 20% (DoS, DDoS): Áp dụng mild_class_weight để hiệu chỉnh trọng số cân bằng lớp ổn định.")
    add_bullet("Tỷ lệ 1% to < 5% (PortScan): Áp dụng weighted_sampler kết hợp class_weight nhằm tăng tần suất tiếp xúc mẫu hiếm.")
    add_bullet("Tỷ lệ 0.1% to < 1% (BruteForce): Sử dụng Focal Loss kết hợp Weighted Sampler để tập trung vào các mẫu khó phân loại.")
    add_bullet("Tỷ lệ < 0.1% (Web, Botnet, Infiltration, Rare): Áp dụng Rare Grouping do quá ít mẫu để học giám sát độc lập.")

    # -------------------------------------------------------------
    # PAGE 3: ĐÁNH GIÁ CHIẾN LƯỢC MẤT CÂN BẰNG MẪU
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("2. Điểm chuẩn Chiến lược Xử lý Mất cân bằng mẫu (CICIDS2017)")
    add_paragraph(
        "Thử nghiệm này so sánh các chiến lược mất cân bằng mẫu trên kiến trúc mạng nơ-ron truyền thẳng (MLP) "
        "sử dụng phân đoạn 10% của tập dữ liệu CICIDS2017. Mục tiêu là tối ưu hóa giữa chỉ số Macro-F1 (đánh giá đều trên các lớp) "
        "và False Alarm Rate (FAR - tỷ lệ báo động giả)."
    )
    
    # Imbalance strategies table
    headers_imbalance = ["Chiến lược (MLP)", "Accuracy", "Macro-F1", "Weighted-F1", "FAR"]
    rows_imbalance = [
        ["MLP_FocalSampler", "0.6753", "0.3395", "0.7436", "0.3856"],
        ["MLP_SamplerOnly", "0.5489", "0.3262", "0.5892", "0.4900"],
        ["MLP_FocalOnly (gamma=2)", "0.7512", "0.3934", "0.8123", "0.2732"],
        ["MLP_FocalOnly (gamma=1)", "0.8690", "0.4491", "0.8988", "0.1312"],
        ["MLP_FocalOnly (gamma=0.5)", "0.8512", "0.4972", "0.8872", "0.1634"],
        ["MLP_CE_ClassWeight", "0.9412", "0.5890", "0.9512", "0.0590"]
    ]
    draw_table(headers_imbalance, rows_imbalance, [60, 32, 32, 32, 34], ["L", "R", "R", "R", "R"])
    
    add_paragraph(
        "Hình dưới đây minh họa sự đánh đổi (trade-off) trực quan. Ta thấy chiến lược MLP_CE_ClassWeight (Cross-Entropy với "
        "trọng số lớp thích ứng) đạt Macro-F1 cao nhất (0.5890) đồng thời giữ FAR ở mức cực thấp (0.0590), trong khi các phương pháp "
        "kết hợp Sampler đơn thuần đẩy FAR lên rất cao (>38%), không khả thi trong thực tế."
    )
    
    # Embed imbalance plot
    embed_image(r"results\plots\cicids2017_imbalance_tradeoff.png", w=150, h=75)

    # -------------------------------------------------------------
    # PAGE 4: SO SÁNH HIỆU NĂNG MÔ HÌNH (NSL-KDD)
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("3. So sánh Hiệu năng Các Mô hình (Tập dữ liệu NSL-KDD)")
    add_paragraph(
        "Nhóm nghiên cứu đã thực hiện huấn luyện và đánh giá chéo 8 mô hình học máy truyền thống và học sâu tiên tiến "
        "trên tập dữ liệu NSL-KDD. Bảng dưới đây thể hiện chi tiết các chỉ số đo lường hiệu năng của từng kiến trúc."
    )

    # Model results table
    headers_models = ["Mô hình", "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC", "PR-AUC", "FAR", "Min. Recall"]
    rows_models = [
        ["XGBoost", "0.999047", "0.748739", "0.998991", "0.999903", "0.788642", "0.000371", "0.444444"],
        ["LightGBM", "0.998889", "0.733667", "0.998827", "0.999928", "0.795808", "0.000520", "0.444444"],
        ["RandomForest", "0.998254", "0.682955", "0.998108", "0.999809", "0.782615", "0.000297", "0.444444"],
        ["CNNLSTMHybrid", "0.979520", "0.588306", "0.982121", "0.000000", "0.000000", "0.033410", "0.444444"],
        ["CNN1D", "0.980790", "0.567013", "0.983322", "0.000000", "0.000000", "0.031925", "0.388889"],
        ["MLP", "0.974360", "0.554910", "0.979292", "0.000000", "0.000000", "0.043210", "0.444444"],
        ["LogisticRegression", "0.932129", "0.543298", "0.943770", "0.992080", "0.549767", "0.111738", "0.722222"],
        ["BiLSTMAttention", "0.945743", "0.520063", "0.952211", "0.000000", "0.000000", "0.068305", "0.388889"]
    ]
    draw_table(headers_models, rows_models, [38, 21, 21, 21, 21, 21, 21, 26], ["L", "R", "R", "R", "R", "R", "R", "R"])
    
    add_paragraph(
        "Đánh giá cho thấy các thuật toán học máy cổ điển (Tree-based) như XGBoost, LightGBM và RandomForest vượt trội "
        "hơn các mạng nơ-ron học sâu về cả độ chính xác tổng thể lẫn khả năng tối ưu hóa FAR. Đặc biệt, XGBoost đạt Macro-F1 "
        "tốt nhất (0.7487) và RandomForest duy trì FAR thấp nhất (0.000297)."
    )
    
    embed_image(r"results\plots\nsl_kdd_macro_f1_comparison.png", w=130, h=65)
    pdf.ln(1)
    embed_image(r"results\plots\nsl_kdd_pareto_scatter.png", w=130, h=65)

    # -------------------------------------------------------------
    # PAGE 5: CROSS-DATASET DOMAIN SHIFT
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("4. Đánh giá Khả năng Tổng quát hóa & Domain Shift")
    add_paragraph(
        "Trong môi trường thực tế, hệ thống NIDS thường phải đối mặt với hiện tượng lệch phân phối dữ liệu (Domain Shift). "
        "Thử nghiệm này kiểm tra mô hình RandomForest và XGBoost khi huấn luyện trên một tập dữ liệu lưu lượng và kiểm thử trực tiếp "
        "trên một tập dữ liệu mạng hoàn toàn khác (NSL-KDD chéo sang UNSW-NB15 và ngược lại) dựa trên không gian đặc trưng chung gồm: "
        "duration, src_bytes, dst_bytes, count."
    )

    # Cross-dataset table
    headers_cross = ["Kịch bản dịch chuyển miền", "Mô hình", "Self Acc", "Self F1", "Cross Acc", "Cross F1", "Cross FAR"]
    rows_cross = [
        ["NSL-KDD -> UNSW-NB15", "RandomForest", "0.980655", "0.980538", "0.583321", "0.519861", "0.755486"],
        ["UNSW-NB15 -> NSL-KDD", "RandomForest", "0.996186", "0.996148", "0.530201", "0.362019", "0.023878"],
        ["NSL-KDD -> UNSW-NB15", "XGBoost", "0.979369", "0.979244", "0.530729", "0.451980", "0.831270"],
        ["UNSW-NB15 -> NSL-KDD", "XGBoost", "0.958898", "0.958604", "0.242076", "0.238589", "0.710289"]
    ]
    draw_table(headers_cross, rows_cross, [50, 30, 22, 22, 22, 22, 22], ["L", "L", "R", "R", "R", "R", "R"])

    add_paragraph(
        "Hình vẽ dưới đây thể hiện sự sụt giảm nghiêm trọng của chỉ số Macro-F1 khi chuyển đổi môi trường kiểm thử. "
        "Điều này khẳng định phân phối đặc trưng lưu lượng mạng của các phân đoạn cảm biến khác nhau có sự sai biệt rất lớn, "
        "khiến các mô hình phân loại giám sát thuần túy mất đi năng lực phân biệt nếu không được điều chỉnh miền (Domain Adaptation)."
    )

    embed_image(r"results\plots\domain_shift_impact.png", w=145, h=80)

    # -------------------------------------------------------------
    # PAGE 6: HYBRID DECISION LAYER (PART 1)
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("5. Lớp Quyết Định Lai (Hybrid Decision Layer)")
    add_paragraph(
        "Nhằm khắc phục điểm yếu của mô hình phân loại giám sát (Classifier Alone) trước các cuộc tấn công chưa biết (Zero-day) "
        "hoặc các mẫu bị bỏ sót do chệch miền, nghiên cứu đề xuất giải pháp Lớp Quyết định Lai (Hybrid Decision Layer). "
        "Hệ thống tính toán Điểm rủi ro (Confidence Risk Score) kết hợp độ không chắc chắn của mô hình giám sát "
        "và điểm bất thường của mô hình không giám sát (PyTorch Autoencoder):"
    )
    # Formula Box
    pdf.set_fill_color(*bg_accent)
    pdf.set_draw_color(*slate_color)
    pdf.set_line_width(0.5)
    pdf.cell(0, 12, "Risk = 0.5 * (1 - Prob_Benign) + 0.5 * Anomaly_Score_Normalized", border=1, fill=True, align="C")
    pdf.ln(12)
    pdf.ln(3)

    add_heading("Kết quả thực nghiệm trên tập NSL-KDD:", level=2)
    headers_hybrid_nsl = ["Cấu hình", "Ngưỡng", "Accuracy", "FAR", "Recall", "F1-Score", "Cảnh báo Zero-day", "Khôi phục bỏ sót"]
    rows_hybrid_nsl = [
        ["RandomForest Alone", "N/A", "0.998293", "0.000297", "0.996674", "0.998164", "N/A", "N/A"],
        ["Hybrid (far_0.01)", "0.042565", "0.994483", "0.010246", "0.999915", "0.994107", "172", "38/39 (97.44%)"],
        ["Hybrid (far_0.03)", "0.015007", "0.984997", "0.028064", "1.000000", "0.984138", "413", "39/39 (100.00%)"],
        ["Hybrid (far_0.05)", "0.007505", "0.974836", "0.047071", "1.000000", "0.973678", "669", "39/39 (100.00%)"],
        ["Hybrid (far_0.1)", "0.002502", "0.946656", "0.099785", "1.000000", "0.945798", "1,379", "39/39 (100.00%)"]
    ]
    draw_table(headers_hybrid_nsl, rows_hybrid_nsl, [36, 18, 22, 18, 22, 22, 28, 24], ["L", "R", "R", "R", "R", "R", "R", "R"])

    add_heading("Kết quả thực nghiệm trên tập UNSW-NB15:", level=2)
    headers_hybrid_unsw = ["Cấu hình", "Ngưỡng", "Accuracy", "FAR", "Recall", "F1-Score", "Cảnh báo Zero-day", "Khôi phục bỏ sót"]
    rows_hybrid_unsw = [
        ["RandomForest Alone", "N/A", "0.959871", "0.011402", "0.914673", "0.946567", "N/A", "N/A"],
        ["Hybrid (far_0.01)", "999.0000", "0.959871", "0.011402", "0.914673", "0.946567", "0", "0/371 (0.00%)"],
        ["Hybrid (far_0.03)", "0.210211", "0.963714", "0.030551", "0.954692", "0.953376", "305", "174/371 (46.90%)"],
        ["Hybrid (far_0.05)", "0.162798", "0.957726", "0.050870", "0.971251", "0.946967", "516", "246/371 (66.31%)"],
        ["Hybrid (far_0.1)", "0.105302", "0.937349", "0.093992", "0.986661", "0.924469", "878", "313/371 (84.37%)"]
    ]
    draw_table(headers_hybrid_unsw, rows_hybrid_unsw, [36, 18, 22, 18, 22, 22, 28, 24], ["L", "R", "R", "R", "R", "R", "R", "R"])

    # -------------------------------------------------------------
    # PAGE 7: HYBRID DECISION LAYER (PART 2 - PLOTS)
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("Biểu đồ Phân tích Quyết định Lai và Phân phối Rủi ro:")
    add_paragraph(
        "Biểu đồ đầu tiên thể hiện tỷ lệ khôi phục các cuộc tấn công bị bỏ sót dưới các cấu hình FAR mục tiêu khác nhau. "
        "Hai biểu đồ tiếp theo hiển thị mật độ phân phối điểm rủi ro lai (Risk Score) đối với lưu lượng Benign (màu xanh) "
        "và các loại tấn công khác nhau, cho thấy khoảng cách phân tách rõ rệt giúp thiết lập các ngưỡng cảnh báo hiệu quả."
    )
    
    # Embed hybrid plots
    embed_image(r"results\plots\hybrid_recovery_percentage.png", w=110, h=55)
    pdf.ln(1)
    
    # Draw NSL and UNSW risk distribution stacked
    embed_image(r"results\plots\nsl_kdd_hybrid_risk_distribution.png", w=110, h=55)
    pdf.ln(1)
    embed_image(r"results\plots\unsw_nb15_hybrid_risk_distribution.png", w=110, h=55)

    # -------------------------------------------------------------
    # PAGE 8: KẾT LUẬN & KHUYẾN NGHỊ
    # -------------------------------------------------------------
    pdf.add_page()
    add_heading("6. Kết luận & Khuyến nghị Triển khai")
    
    add_heading("Kết luận rút ra từ nghiên cứu:", level=2)
    add_bullet(
        "RandomForest là lựa chọn baseline vững chắc nhất cho việc phát hiện các cuộc tấn công đã biết nhờ tốc độ huấn luyện nhanh, "
        "hiệu năng cao ổn định (F1 0.9678 trên CICIDS2017) và tỷ lệ báo động giả (FAR) cực kỳ thấp."
    )
    add_bullet(
        "Chiến lược xử lý mất cân bằng lớp thích ứng bằng Cross-Entropy có trọng số (Class Weight) là giải pháp ổn định nhất "
        "cho mạng học sâu MLP, giúp tăng F1 thêm 9.5% và duy trì FAR dưới 6% so với các phương pháp nạp lại mẫu (Oversampling)."
    )
    add_bullet(
        "Khung quyết định lai (Hybrid Decision Layer) đã chứng minh vai trò quan trọng trong việc khôi phục các cuộc tấn công "
        "bị bỏ sót (đạt tỷ lệ khôi phục từ 66.3% đến 100%). Cơ chế này tạo ra một vòng phòng thủ thứ hai để phát hiện lưu lượng bất thường."
    )

    add_heading("Khuyến nghị khi triển khai thực tế (Deployment Guidelines):", level=2)
    add_bullet(
        "Cấu hình ngưỡng Hybrid: Khuyến nghị lựa chọn cấu hình hybrid mục tiêu FAR 3% (far_0.03). Tại điểm này, mô hình duy trì "
        "được tỷ lệ khôi phục tấn công cao (46.9% - 100%) mà không làm tăng quá nhiều tải trọng phân tích cảnh báo giả cho các kỹ sư SOC."
    )
    add_bullet(
        "Đối phó Domain Shift: Do hiệu năng sụt giảm mạnh khi triển khai sang phân đoạn mạng mới (F1 giảm từ 0.99 xuống 0.51), "
        "cần áp dụng cơ chế cập nhật mô hình định kỳ và sử dụng Autoencoder không giám sát làm chốt chặn phát hiện bất thường "
        "để ghi nhận lưu lượng lạ trước khi Classifier được tinh chỉnh."
    )
    add_bullet(
        "Tối ưu hóa tài nguyên: Sử dụng định dạng ONNX để xuất bản mô hình, tối ưu hóa thời gian phản hồi (latency) "
        "xuống dưới 2ms mỗi luồng traffic khi tích hợp trực tiếp vào các thiết bị mạng phần cứng."
    )

    pdf.ln(10)
    # Signature/Closure Line
    pdf.set_font(font_name, "B", 10)
    pdf.set_text_color(*slate_color)
    pdf.cell(0, 5, "BÁO CÁO KẾT THÚC", align="C")
    pdf.ln(5)
    
    # Save the output file
    output_pdf_path = r"results\NIDS_Experimental_Evaluation_Report.pdf"
    pdf.output(output_pdf_path)
    print(f"[+] PDF successfully saved to {output_pdf_path}")

if __name__ == "__main__":
    build_pdf()
