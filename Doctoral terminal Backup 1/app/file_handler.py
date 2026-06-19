import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

class FileHandler:
    """Handle file uploads, validation, and storage"""
    
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf', 'dcm'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, upload_folder='app/static/uploads'):
        self.upload_folder = upload_folder
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Create upload directory if it doesn't exist"""
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    def _ensure_patient_dir(self, patient_id):
        """Create patient-specific directory if it doesn't exist"""
        patient_dir = os.path.join(self.upload_folder, str(patient_id))
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        return patient_dir
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def validate_file(self, file):
        """Validate file type and size"""
        if not file:
            return False, "No file provided"
        
        if file.filename == '':
            return False, "No file selected"
        
        if not self.allowed_file(file.filename):
            return False, f"File type not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Check file size by seeking to end
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB"
        
        return True, "Valid"
    
    def save_uploaded_file(self, file, patient_id, file_type='general'):
        """
        Save uploaded file to patient directory
        
        Args:
            file: FileStorage object from Flask request
            patient_id: Patient ID
            file_type: Type of file (lab, xray, timeline, etc.)
        
        Returns:
            dict: File metadata including path, url, size, etc.
        """
        # Validate file
        is_valid, message = self.validate_file(file)
        if not is_valid:
            return {"error": message}
        
        # Ensure patient directory exists
        patient_dir = self._ensure_patient_dir(patient_id)
        
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        new_filename = f"{file_type}_{timestamp}_{unique_id}.{file_ext}"
        
        # Save file
        file_path = os.path.join(patient_dir, new_filename)
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Generate URL path (relative to static folder)
        url_path = f"/static/uploads/{patient_id}/{new_filename}"
        
        return {
            "success": True,
            "filename": new_filename,
            "original_filename": original_filename,
            "file_path": file_path,
            "url": url_path,
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "uploaded_at": datetime.now().isoformat(),
            "file_type": file_type
        }
    
    def delete_file(self, patient_id, filename):
        """Delete a file from patient directory"""
        file_path = os.path.join(self.upload_folder, str(patient_id), filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def get_file_url(self, patient_id, filename):
        """Generate URL for file access"""
        return f"/static/uploads/{patient_id}/{filename}"
    
    def get_file_path(self, patient_id, filename):
        """Get absolute file path"""
        return os.path.join(self.upload_folder, str(patient_id), filename)

# Global instance
file_handler = FileHandler()
